import os

from label_studio_converter import brush
from typing import List, Dict, Optional
from uuid import uuid4
from .sam_predictor import PredictorPool
from mdi_sam_server.label_studio_ml_mdi.model import LabelStudioMLBase

SAM_CHOICE = os.environ.get("SAM_CHOICE", "SAM")
# 实例池大小：建议设置为 GPU 数量（多 GPU 并行）或单 GPU 上允许同时加载的模型副本数。
# 每增加 1 个实例约多占一份模型显存（SAM2-base+ ~2-4 GB VRAM）。
POOL_SIZE = int(os.environ.get("PREDICTOR_POOL_SIZE", "1"))

PREDICTOR_POOL = PredictorPool(SAM_CHOICE, pool_size=POOL_SIZE)


class SamMLBackend(LabelStudioMLBase):

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Returns the predicted mask for a smart keypoint that has been placed."""

        if not context or not context.get('result'):
            # if there is no context, no interaction has happened yet/home/mdi/zhangcheng-dev/mdi-sam/sam_server/sam_backend
            return []
        
        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']
        layer_cur_scale = context.get('layer_cur_scale',None)

        # collect context information
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None
        for ctx in context['result']:
            x = ctx['value']['x'] * image_width / 100
            y = ctx['value']['y'] * image_height / 100
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]
            if ctx_type == 'keypointlabels':
                point_labels.append(int(ctx['is_positive']))
                point_coords.append([int(x), int(y)])
            elif ctx_type == 'rectanglelabels':
                box_width = ctx['value']['width'] * image_width / 100
                box_height = ctx['value']['height'] * image_height / 100
                input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]

        #print(f'Point coords are {point_coords}, point labels are {point_labels}, input box is {input_box}')

        img_path = tasks[0]['data']['image']
        with PREDICTOR_POOL.acquire() as predictor:
            predictor_results = predictor.predict(
                img_path=img_path,
                point_coords=point_coords or None,
                point_labels=point_labels or None,
                input_box=input_box
            )

        predictions = self.get_results(
            masks=predictor_results['masks'],
            probs=predictor_results['probs'],
            bbox=predictor_results['bbox'],
            cur_scale=layer_cur_scale,
            width=image_width,
            height=image_height,
            label=selected_label)

        return predictions

    def get_results(self, masks, probs, width, height, bbox, label,cur_scale):
        results = []
        for mask, prob in zip(masks, probs):
            # creates a random ID for your label everytime so no chance for errors
            label_id = str(uuid4())[:4]
            # converting the mask from the model to RLE format which is usable in Label Studio
            mask = mask * 255
            rle = brush.mask2rle(mask)

            results.append({
                'id': label_id,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'layer_cur_scale':cur_scale,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'bbox': bbox,
                    'brushlabels': [label],
                },
                'score': prob,
                'type': 'brushlabels',
                'readonly': False
            })

        return [{
            'result': results,
            'model_version': PREDICTOR_POOL.model_name
        }]

    def preload(self, url):
        if url:
            try:
                with PREDICTOR_POOL.acquire() as predictor:
                    predictor.set_image(url, calculate_embeddings=False)
            except Exception as err:
                print(f'error info :{err}')
                raise Exception(f"error info: preload faild!")

if __name__ == '__main__':
    # test the model
    model = SamMLBackend()

    model.use_label_config("")
    results = model.predict(
        tasks=[{
            'data': {
                'image': 'https://s3.amazonaws.com/htx-pub/datasets/images/125245483_152578129892066_7843809718842085333_n.jpg'
            }}],
        context={
            'result': [{
                'original_width': 1080,
                'original_height': 1080,
                'image_rotation': 0,
                'value': {
                    'x': 49.441786283891545,
                    'y': 59.96810207336522,
                    'width': 0.3189792663476874,
                    'keypointlabels': ['Banana']
                },
                'is_positive': True,
                'type': 'keypointlabels',
                'origin': 'manual'
            }]}
    )
    import json
    results[0]['result'][0]['value']['rle'] = f'...{len(results[0]["result"][0]["value"]["rle"])} integers...'
    print(json.dumps(results, indent=2))
