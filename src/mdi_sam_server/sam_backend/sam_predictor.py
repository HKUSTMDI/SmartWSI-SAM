import os
import logging
import queue
import threading
from contextlib import contextmanager
import torch
import cv2
import numpy as np

from typing import List, Dict, Optional
from mdi_sam_server.label_studio_ml_mdi.utils import get_image_local_path, InMemoryLRUDictCache

import matplotlib.pyplot as plt
import uuid

logger = logging.getLogger(__name__)

VITH_CHECKPOINT = os.environ.get("VITH_CHECKPOINT", "../models/sam_vit_l_0b3195.pth")
VITH_REG_KEY = os.environ.get("VITH_REG_KEY", "vit_l")
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "../models/sam2_hiera_base_plus.pt")
SAM2_CONFIG = os.environ.get("SAM2_CONFIG", "sam2_hiera_b+.yaml")
ONNX_CHECKPOINT = os.environ.get("ONNX_CHECKPOINT", "../models/sam_onnx_quantized_example.onnx")
MOBILESAM_CHECKPOINT = os.environ.get("MOBILESAM_CHECKPOINT", "../models/mobile_sam.pt")
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")

SAM_DRAW_MODE=os.environ.get("SAM_DRAW_MODE",False)


class SAMPredictor(object):

    def __init__(self, model_choice, device: str = None):
        self.model_choice = model_choice

        # cache for embeddings
        # TODO: currently it supports only one image in cache,
        #   since predictor.set_image() should be called each time the new image comes
        #   before making predictions
        #   to extend it to >1 image, we need to store the "active image" state in the cache
        self.cache = InMemoryLRUDictCache(1)

        # 每个实例独立运行在指定设备上，PredictorPool 保证同一时刻只有一个线程持有此实例，
        # 因此不需要内部 Lock。
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device {self.device}")

        if model_choice == 'ONNX':
            import onnxruntime
            from segment_anything import sam_model_registry, SamPredictor

            self.model_checkpoint = VITH_CHECKPOINT
            if self.model_checkpoint is None:
                raise FileNotFoundError("VITH_CHECKPOINT is not set: please set it to the path to the SAM checkpoint")
            if ONNX_CHECKPOINT is None:
                raise FileNotFoundError("ONNX_CHECKPOINT is not set: please set it to the path to the ONNX checkpoint")
            logger.info(f"Using ONNX checkpoint {ONNX_CHECKPOINT} and SAM checkpoint {self.model_checkpoint}")

            self.ort = onnxruntime.InferenceSession(ONNX_CHECKPOINT)
            reg_key = VITH_REG_KEY

        elif model_choice == 'SAM':
            from segment_anything import SamPredictor, sam_model_registry

            self.model_checkpoint = VITH_CHECKPOINT
            if self.model_checkpoint is None:
                raise FileNotFoundError("VITH_CHECKPOINT is not set: please set it to the path to the SAM checkpoint")

            logger.info(f"Using SAM checkpoint {self.model_checkpoint}")
            reg_key = VITH_REG_KEY

        elif model_choice == 'SAM2':
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self.model_checkpoint = SAM2_CHECKPOINT
            if self.model_checkpoint is None:
                raise FileNotFoundError("SAM2_CHECKPOINT is not set: please set it to the path to the SAM2 checkpoint")

            logger.info(f"Using SAM2 checkpoint {self.model_checkpoint}")
            model_cfg = SAM2_CONFIG
            sam2_model = build_sam2(model_cfg, self.model_checkpoint, device=self.device)
            
            self.predictor = SAM2ImagePredictor(sam2_model)
            return
            
        elif model_choice == 'MobileSAM':
            from mobile_sam import SamPredictor, sam_model_registry

            self.model_checkpoint = MOBILESAM_CHECKPOINT
            if not self.model_checkpoint:
                raise FileNotFoundError("MOBILE_CHECKPOINT is not set: please set it to the path to the MobileSAM checkpoint")
            logger.info(f"Using MobileSAM checkpoint: {self.model_checkpoint}")
            reg_key = 'vit_t'
        else:
            raise ValueError(f"Invalid model choice {model_choice}")

        sam = sam_model_registry[reg_key](checkpoint=self.model_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    @property
    def model_name(self):
        return f'{self.model_choice}:{self.model_checkpoint}:{self.device}'

    def set_image(self, img_path, calculate_embeddings=True):
        payload = self.cache.get(img_path)
        print("payload:",payload)
        if payload is None:
            # Get image and embeddings
            logger.info(f'Payload not found for {img_path} in `IN_MEM_CACHE`: calculating from scratch')
            image_path = get_image_local_path(
                img_path,
                label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                label_studio_host=LABEL_STUDIO_HOST
            )
            logger.info(f"image_path:{image_path}")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)

            payload = {'image_shape': image.shape[:2]}
            logger.debug(f'Finished set_image({img_path}) in `IN_MEM_CACHE`: image shape {image.shape[:2]}')
            if calculate_embeddings:
                image_embedding = self.predictor.get_image_embedding().cpu().numpy()
                payload['image_embedding'] = image_embedding
                logger.debug(f'Finished storing embeddings for {img_path} in `IN_MEM_CACHE`: '
                             f'embedding shape {image_embedding.shape}')
            self.cache.put(img_path, payload)
        else:
            logger.debug(f"Using embeddings for {img_path} from `IN_MEM_CACHE`")
        return payload

    def predict_onnx(
        self,
        img_path,
        point_coords: Optional[List[List]] = None,
        point_labels: Optional[List] = None,
        input_box: Optional[List] = None
    ):
        # calculate embeddings
        payload = self.set_image(img_path, calculate_embeddings=True)
        image_shape = payload['image_shape']
        image_embedding = payload['image_embedding']

        onnx_point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        onnx_point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        onnx_box_coords = np.array(input_box, dtype=np.float32).reshape(2, 2) if input_box else None

        onnx_coords, onnx_labels = None, None
        if onnx_point_coords is not None and onnx_box_coords is not None:
            # both keypoints and boxes are present
            onnx_coords = np.concatenate([onnx_point_coords, onnx_box_coords], axis=0)[None, :, :]
            onnx_labels = np.concatenate([onnx_point_labels, np.array([2, 3])], axis=0)[None, :].astype(np.float32)

        elif onnx_point_coords is not None:
            # only keypoints are present
            onnx_coords = np.concatenate([onnx_point_coords, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_labels = np.concatenate([onnx_point_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)

        elif onnx_box_coords is not None:
            # only boxes are present
            raise NotImplementedError("Boxes without keypoints are not supported yet")

        onnx_coords = self.predictor.transform.apply_coords(onnx_coords, image_shape).astype(np.float32)

        # TODO: support mask inputs
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)

        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coords,
            "point_labels": onnx_labels,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image_shape, dtype=np.float32)
        }

        masks, prob, low_res_logits = self.ort.run(None, ort_inputs)
        masks = masks > self.predictor.model.mask_threshold
        mask = masks[0, 0, :, :].astype(np.uint8)  # each mask has shape [H, W]
        prob = float(prob[0][0])
        # TODO: support the real multimask output as in https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
        return {
            'masks': [mask],
            'probs': [prob]
        }

    def predict_sam(
        self,
        img_path,
        point_coords: Optional[List[List]] = None,
        point_labels: Optional[List] = None,
        input_box: Optional[List] = None
    ):

        self.set_image(img_path, calculate_embeddings=False)
        point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(input_box, dtype=np.float32) if input_box else None
        logger.debug("predict...")
        masks, probs, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            # TODO: support multimask output
            multimask_output=False
        )
        mask = masks[0].astype(np.uint8)  # each mask has shape [H, W]

        # test the result, and draw the mask on image
        points = point_coords
        if SAM_DRAW_MODE:
            logger.debug("drawing...")
            self.show_mask(points, point_labels, mask, img_path, bbox= input_box)

        # 计算轮廓
        logger.debug("Calculate the external profile")
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 计算外接矩形
        new_contours = []
        for contour in contours:
            new_contours.extend(list(contour))
        new_contours = np.array(new_contours)
        x, y, w, h = cv2.boundingRect(new_contours)
        bbox = [x, y, w, h]
        
        logger.debug("mask_info:")
        prob = float(probs[0])
        return {
            'masks': [mask],
            'probs': [prob],
            'bbox' : bbox
        }

    def predict(
        self, img_path: str,
        point_coords: Optional[List[List]] = None,
        point_labels: Optional[List] = None,
        input_box: Optional[List] = None
    ):
        if self.model_choice == 'ONNX':
            return self.predict_onnx(img_path, point_coords, point_labels, input_box)
        elif self.model_choice in ('SAM', 'MobileSAM', 'SAM2'):
            return self.predict_sam(img_path, point_coords, point_labels, input_box)
        else:
            raise NotImplementedError(f"Model choice {self.model_choice} is not supported yet")


    def show_mask(self, points, labels, mask, image_path , bbox=None, random_color=True , local_test=False):
        """
        mask画图
        """
        import matplotlib.patches as mpatches
        if not local_test:
            image_path = get_image_local_path(
                    image_path,
                    label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                    label_studio_host=LABEL_STUDIO_HOST
                )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])

        mask_image_array = np.array(mask)
        h, w = mask_image_array.shape[-2:]
        mask_image_ins = mask_image_array.reshape(h, w, 1) * color.reshape(1, 1, -1)
        id = str(uuid.uuid4())[:6]
        #画图
        plt.figure(figsize=(10,10))
        #画标注点
        if labels is not None:
            pos_points = points[labels==1]
            neg_points = points[labels==0]
            plt.scatter(pos_points[:,0],pos_points[:,1], s=150, color='green',marker='*')
            plt.scatter(neg_points[:,0],neg_points[:,1], s=150, color='red',marker='*')
        #画框
        if bbox is not None:
            x0 = bbox[0]
            y0 = bbox[1]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            plt.gca().add_patch(mpatches.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

        plt.imshow(image)
        plt.imshow(mask_image_ins)
        #plt.axis('off')
        
        if not os.path.exists('../draw_image/'):
            os.mkdir('../draw_image/')

        plt.savefig(f"../draw_image/mask_{id}")



class PredictorPool:
    """
    固定大小的 SAMPredictor 实例池，支持多 GPU 并行推理。

    工作原理：
    - 启动时按 pool_size 创建 N 个 SAMPredictor，按 round-robin 分配到各 GPU。
    - 请求通过 acquire() 从池中取出一个空闲实例（若全忙则阻塞等待）。
    - 推理完成后 contextmanager 自动归还，下一个等待请求立即获得它。
    - 每个实例状态完全独立，无需 Lock，真正并行。

    单 GPU 场景：
    - 多实例在同一 GPU 上，现代显卡（A100/H100）通常能容纳 2-4 个 SAM2 实例。
    - 当一个实例在做 encode（GPU 密集），另一个可同时完成 I/O + decode（GPU 利用率更高）。

    多 GPU 场景：
    - pool_size >= gpu_count 时每个 GPU 至少分配 1 个实例，实现真正的跨设备并行。
    """

    def __init__(self, model_choice: str, pool_size: int):
        gpu_count = torch.cuda.device_count()
        logger.info(f"PredictorPool: pool_size={pool_size}, visible_gpus={gpu_count}")

        self._pool: queue.Queue = queue.Queue()
        self._model_name: str = ""

        for i in range(pool_size):
            if gpu_count > 0:
                device = f"cuda:{i % gpu_count}"
            else:
                device = "cpu"
            logger.info(f"  Loading predictor [{i}] on {device} ...")
            predictor = SAMPredictor(model_choice, device=device)
            if i == 0:
                self._model_name = predictor.model_name
            self._pool.put(predictor)

        logger.info(f"PredictorPool ready: {pool_size} instance(s), model={self._model_name}")

    @contextmanager
    def acquire(self):
        """从池中取出一个空闲 predictor，用完后自动归还。"""
        predictor = self._pool.get()
        try:
            yield predictor
        finally:
            self._pool.put(predictor)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def size(self) -> int:
        return self._pool.qsize()


if __name__ == "__main__":
    predicater = SAMPredictor('SAM')
    input_point = np.array([[500, 375],[500,475]])
    input_box = [500, 375, 600, 475]
    input_label = np.array([1,0])
    mask        = np.array([[1, 1]])

    url = './test_image/truck.jpg'
    predicater.show_mask(input_point, input_label, mask, url, bbox=input_box, local_test=True)