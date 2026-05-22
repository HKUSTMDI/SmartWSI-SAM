import logging

from PIL import Image, ImageOps, ImageColor
from collections import OrderedDict

from label_studio_tools.core.utils.params import get_env
from label_studio_tools.core.utils.io import get_local_path

from typing import List, Dict, Optional
from .conf.settings import CONFIG
import requests
import math
import re
import os
import concurrent.futures
from io import BytesIO
import threading
import os
import aiohttp
import asyncio
import time
import hashlib
import urllib
from functools import wraps
from appdirs import user_data_dir

DATA_UNDEFINED_NAME = '$undefined$'

logger = logging.getLogger(__name__)
_DIR_APP_NAME = 'label-studio'

LOCAL_FILES_DOCUMENT_ROOT = get_env(
    'LOCAL_FILES_DOCUMENT_ROOT', default=os.path.abspath(os.sep)
)
def get_data_dir():
    data_dir = user_data_dir(appname=_DIR_APP_NAME)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def cost_time(func):
    @wraps(func)
    def decrated(*args, **kwargs):
        start_time = time.time()
        result = func(*args,**kwargs)
        cost_time = round(time.time() - start_time, 3) * 1000
        logger.debug(f"[function:{func.__name__}] cost {cost_time} ms")
        return result
    return decrated


def get_single_tag_keys(parsed_label_config, control_type, object_type):
    """
    Gets parsed label config, and returns data keys related to the single control tag and the single object tag schema
    (e.g. one "Choices" with one "Text")
    :param parsed_label_config: parsed label config returned by "label_studio.misc.parse_config" function
    :param control_type: control tag str as it written in label config (e.g. 'Choices')
    :param object_type: object tag str as it written in label config (e.g. 'Text')
    :return: 3 string keys and 1 array of string labels: (from_name, to_name, value, labels)
    """
    assert len(parsed_label_config) == 1
    from_name, info = list(parsed_label_config.items())[0]
    assert info['type'] == control_type, 'Label config has control tag "<' + info['type'] + '>" but "<' + control_type + '>" is expected for this model.'  # noqa

    assert len(info['to_name']) == 1
    assert len(info['inputs']) == 1
    assert info['inputs'][0]['type'] == object_type
    to_name = info['to_name'][0]
    value = info['inputs'][0]['value']
    return from_name, to_name, value, info['labels']


def get_first_tag_keys(parsed_label_config, control_type, object_type):
    """
    Reads config and returns the first control tag and the first object tag that match the given types
    :param parsed_label_config:
    :param control_type:
    :param object_type:
    :return:
    """
    for from_name, info in parsed_label_config.items():
        if info['type'] == control_type:
            for input in info['inputs']:
                if input['type'] == object_type:
                    return from_name, info
    return None, None


def is_skipped(completion):
    if len(completion['annotations']) != 1:
        return False
    completion = completion['annotations'][0]
    return completion.get('skipped', False) or completion.get('was_cancelled', False)


def get_choice(completion):
    return completion['annotations'][0]['result'][0]['value']['choices'][0]

def get_image_local_path(url, image_cache_dir=None, project_dir=None, image_dir=None,
                         label_studio_host=None, label_studio_access_token=None):
    
    is_local_file = url.startswith('/data/') and '?d=' in url
    is_uploaded_file = url.startswith('/data/upload')
    if image_dir is None:
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        image_dir = project_dir and os.path.join(project_dir, 'upload') or upload_dir
        
    # File reference created with --allow-serving-local-files option
    if is_local_file:
        filename, dir_path = url.split('/data/', 1)[-1].split('?d=')
        dir_path = str(urllib.parse.unquote(dir_path))
        filepath = os.path.join(LOCAL_FILES_DOCUMENT_ROOT, dir_path)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        filepath = os.path.join(filepath,filename)
        
        return filepath
    
    image_local_path = get_local_path(
        url=url,
        cache_dir=image_cache_dir,
        project_dir=project_dir,
        hostname=label_studio_host or get_env('HOSTNAME'),
        image_dir=image_dir,
        access_token=label_studio_access_token
    )
    logger.info(f'Image stored in the local path: {image_local_path}')
    return image_local_path


def get_image_size(filepath):
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    return img.size

@cost_time
async def download_image(session, image):
    url = image['tile_url']
    max_retry = 3
    retry_count = 0

    while retry_count < max_retry:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data         = await response.read()
                    image_ins          = Image.open(BytesIO(image_data))
                    image['content']   = image_ins
                    image['width']     = image_ins.size[0]
                    image['height']     = image_ins.size[1]
                    #存储瓦片信息(debug使用)
                    #local_tile_storage = test_tile_storage + image['file_name'] + ".jpg"
                    #image['content'].save(local_tile_storage)
                else:
                    logger.error("Failed to download image")
                    raise Exception("Failed to download image")
            # 下载成功则跳出循环
            break
        except Exception as e:
            # 失败时进行延时并增加重试计数
            retry_count += 1
            if retry_count < max_retry:
                delay_seconds = 2  # 延时时间（秒）
                await asyncio.sleep(delay_seconds)
                continue
            else:
                raise e
        
class InMemoryLRUDictCache:
    def __init__(self, capacity=1):
        self.cache = OrderedDict()
        self.capacity = capacity

    def __contains__(self, item):
        return item in self.cache

    def get(self, key):
        if key in self.cache:
            # Move the accessed item to the end
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            # Move the updated item to the end
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Pop the first item if cache reached its capacity
            self.cache.popitem(last=False)

        self.cache[key] = value

    def __str__(self):
        return str(self.cache)


class wsiHandler:
    def __init__(self) -> None:
        self.cache = InMemoryLRUDictCache(50)

    def get_cache_urlInfo(self, image_info_url):
        if not image_info_url in self.cache:
            logger.info(f"no info in cache,ready to request:{image_info_url}")
            start_time = time.time()
            resp = requests.get(image_info_url)
            if resp.status_code == 200:
                image_info = resp.json()
                self.cache.put(image_info_url,image_info)
            else:
                logger.debug(f"get sdpc image[{image_info_url}]info faild,status code:[{resp.status_code}]")
                raise Exception(f"get sdpc image[{image_info_url}]info faild,status code:[{resp.status_code}]")
            cost_time = round((time.time() - start_time) * 1000,3)
            logger.info(f"request url_info cost_time:{cost_time} ms")
        else:
            logger.info("use image_info_url in cache")
            image_info = self.cache.get(image_info_url)

        return image_info
    
    @cost_time
    def create_image(self, tile_size, slice_size_num, layer, point_list:list, prefix_url, image_filename):
        """
        description:tile瓦片拼接出图片
        """
        local_slide_exist = False
        url_layer = layer - 1
        image_url_list= []

        #判断本地是否存在slice的拼接图
        hash_name = hashlib.sha256(str(point_list).encode('utf-8')).hexdigest()
        slice_filename = image_filename  + '_' + str(layer) + '_' + hash_name + '.jpeg'
        local_slice_filename = os.path.join(CONFIG.local_storage, slice_filename)

        if os.path.exists(local_slice_filename):
            logger.info(f"local storage exist slide:{local_slice_filename}")
            local_slide_exist = True #本地已存在该slide 
        
        for item in point_list:
            logger.debug(f"prefix_url:{prefix_url}")
            tile_url = prefix_url + '/' + str(url_layer) + '/' + str(item[0]) + '/' + str(item[1])
            file_name = image_filename + '_'  + str(url_layer) + '_' + str(item[0]) + '_' + str(item[1])
            logger.debug(f"tile_url:[{tile_url}],file_name:[{file_name}]")

            image_url_list.append(
                {
                    "tile_url"      : tile_url,
                    "file_name"     : file_name,
                    "layer"         : layer,
                    "x"             : item[0],
                    "y"             : item[1],
                    "content"       : None,
                    file_name       : True
                }
            )
        #download_images(image_url_list)
        tasks = []
        logger.debug(f"image_url_list:{image_url_list}")
        start_time = time.time()
        async def process_urls():
            async with aiohttp.ClientSession() as session:
                for url in image_url_list:
                    task = asyncio.ensure_future(download_image(session, url))
                    tasks.append(task)
                await asyncio.gather(*tasks)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(process_urls())
        loop.close()
        cost_time = round((time.time() - start_time) * 1000,3)
        logger.info(f"download tile images cost time:{cost_time}ms")

        #计算拼接slice图的宽高
        slice_width  = 0
        slice_height = 0
        for item, image in enumerate(image_url_list):
            if item // slice_size_num[0] == 0:
                slice_width += image['width']
            if item % slice_size_num[0] == 0:
                slice_height += image['height']

        logger.debug(f"slice_size_num:{slice_size_num}")
        logger.debug(f"new slice size:[{slice_width},{slice_height}]")

        slice_image = Image.new('RGB', (slice_width, slice_height), ImageColor.getrgb("white"))
        logger.debug(f"image_url_list:{image_url_list}")

        #本地不存在则拼接出该slide
        if not local_slide_exist:
            for item, image in enumerate(image_url_list):
                slice_x = int(item % slice_size_num[0] * tile_size[0])
                slice_y = int(item // slice_size_num[0] * tile_size[1])
                slice_image.paste(image["content"], (slice_x, slice_y))
            
            #url文件名:/data/5cf58b__CMWGTUhghiTnTpwd.jpg?d=home/mdi/.cache/label-studio;
            #本地文件名:/home/mdi/.cache/label-studio/5cf58b__CMWGTUhghiTnTpwd.jpg
            #保存在本地，本地文件名:原文件名+时间戳(ms)
            os.makedirs(CONFIG.local_storage, exist_ok=True)
            slice_image.save(local_slice_filename)
        url_slice_filename = "/data/" + slice_filename + '?d=' + CONFIG.local_storage

        return url_slice_filename, slice_width, slice_height
    

    def sdpc_convert(self, tasks:List[Dict], context: Optional[Dict] = None, **kwargs):
        """
        Description: sdpc实时sam识别
        """
        #获取图片信息，请求CONFIG.sdpc_tile_prefix
        image_info_url = tasks[0]['data']['image']
        pattern = re.escape(CONFIG.sdpc_tile_prefix)
        image_filename = re.sub(pattern, '', image_info_url)

        #获取wsi信息
        image_info = self.get_cache_urlInfo(image_info_url)
        #计算layer位置
        cur_scale        = context['cur_scale']
        layer_size       = image_info['basisInfo']['layerSize']
        tile_width       = image_info['basisInfo']['tileWidth']
        tile_height      = image_info['basisInfo']['tileHeight']
        sliceLayerInfo   = image_info['originalInfo']['sliceLayerInfo']
        current_layer    = self.get_layer_level(sliceLayerInfo, cur_scale)
        
        logger.debug(f"current_layer:current_layer")
        current_layer_info = sliceLayerInfo[current_layer-1]
        layer_cur_scale  = current_layer_info['curScale']

        #拼接、保存图片
        current_layer_sliceX = current_layer_info['sliceNumX']
        current_layer_sliceY = current_layer_info['sliceNumY']
        logger.info(f"current_layer_info:{current_layer_info},layer_cur_scale:{layer_cur_scale}")
        rectangle_data = None

        #获取rectangle prompt信息
        for ctx in context['result']:
            if ctx["type"] == "rectanglelabels":
                rectangle_data = ctx
                break
        if rectangle_data == None:
            logger.error("no rectangle prompt!please check")
            raise Exception("no rectagnle prompt!")
        
        #通过rectangle矩形框获取tile位置
        input_data  =  rectangle_data
        point_x     = input_data['value']['x'] / 100
        point_y     = input_data['value']['y'] / 100
        box_width   = input_data['value']['width'] / 100 
        box_height  = input_data['value']['height'] / 100
        
        logger.debug(f"point_x:{point_x},point_y:{point_y},box_width:{box_width},box_height:{box_height}")
        #左上角tile瓦片位置
        layer_x_min = math.floor( point_x * current_layer_sliceX ) #计算所需要的layer块x方向的位置
        layer_y_min = math.floor( point_y * current_layer_sliceY ) #计算所需要的layer块y方向的位置
        layer_x_max = math.floor( (point_x + box_width) * current_layer_sliceX )
        layer_y_max = math.floor( (point_y + box_height) * current_layer_sliceY )

        slice_width_num  = layer_x_max - layer_x_min + 1
        slice_height_num = layer_y_max - layer_y_min + 1
        tile_size        = ( tile_width , tile_height )
        slice_size_num   = ( slice_width_num, slice_height_num )

        logger.debug(f"layer_x_min:{layer_x_min},layer_x_max:{layer_x_max}")
        logger.debug(f"layer_y_min:{layer_y_min},layer_y_max:{layer_y_max}")
        box_point_list = []
        for i in range(layer_y_min, layer_y_max + 1):
            for j in range(layer_x_min, layer_x_max + 1):
                box_point_list.append([j,i])

        logger.debug(f"box_point_list:{box_point_list}")
        #图片下载、拼接
        #tile_image_url:瓦片图前缀
        tile_image_url = CONFIG.sdpc_tile_imageURL + image_filename
        #判断图片收集否存在(优化)
        slice_url, slice_width, slice_height = self.create_image(tile_size, slice_size_num, current_layer, 
                                      box_point_list, tile_image_url, image_filename)
        
        #修改tasks,contex的值,x,y,width,height都要改
        tasks[0]['data']['image']     = slice_url
        input_data['original_width']  = slice_width
        input_data['original_height'] = slice_height
        

        #对prompt中的坐标进行转换:
        #x,y,width,height位置变换:在拼接的slice中，输入点相对于参考点的坐标位置
        context['layer_cur_scale'] = layer_cur_scale
        for ctx in context['result']:
            point_x_ins = ctx['value']['x'] / 100
            point_y_ins = ctx['value']['y'] / 100
            ctx['value']['x'] = (point_x_ins * current_layer_sliceX - layer_x_min) / slice_width_num * 100
            ctx['value']['y'] = (point_y_ins * current_layer_sliceY - layer_y_min) / slice_height_num * 100

            logger.debug(f"ctx['type']:{ctx['type']}, x_relative:{ctx['value']['x']},y_relative:{ctx['value']['y']}")
                         
            if ctx['type'] == "rectanglelabels":
                box_width_ins  = ctx['value']['width'] / 100
                box_height_ins = ctx['value']['height'] / 100
                ctx['value']['width']   = box_width_ins * current_layer_sliceX / slice_width_num * 100
                ctx['value']['height']  = box_height_ins * current_layer_sliceY / slice_height_num * 100
                
                logger.debug(f"box_width_relative:{ctx['value']['width']}, box_height_relative:{ctx['value']['height']}")
                           
    def svs_handler(self, tasks:List[Dict], context: Optional[Dict] = None, **kwargs):
        """
        Description: svs类型SAM实时识别,兼容tiff格式图片
        """
        #获取图片信息，请求svs_tile_prefix
        image_info_url = tasks[0]['data']['image']
        pattern = re.escape(CONFIG.svs_tile_prefix)
        image_filename = re.sub(pattern, '', image_info_url)
        #wsi信息
        image_info = self.get_cache_urlInfo(image_info_url)

        #计算layer位置
        cur_scale        = context['cur_scale']
        layer_size       = image_info['basisInfo']['layerSize']
        tile_width       = int(image_info['basisInfo']['tileWidth'])
        tile_height      = int(image_info['basisInfo']['tileHeight'])
        sliceLayerInfo   = image_info['sliceInfo'] #这里和sdpc略有不同
        current_layer    = self.get_layer_level(sliceLayerInfo, cur_scale)
        logger.debug(f"current_layer:{current_layer}")
        
        current_layer_info = sliceLayerInfo[current_layer-1]
        layer_cur_scale  = current_layer_info['curScale']
        #拼接、保存图片
        current_layer_sliceX = current_layer_info['sliceNumX']
        current_layer_sliceY = current_layer_info['sliceNumY']
        current_layer_width  = current_layer_info['sliceWidth']
        current_layer_height = current_layer_info['sliceHeight']
        logger.debug("current_layer_info:",current_layer_info)
        
        #获取rectangle prompt信息
        rectangle_data = None
        for ctx in context['result']:
            if ctx["type"] == "rectanglelabels":
                rectangle_data = ctx
                break
        if rectangle_data == None:
            logger.error("no rectangle prompt!please check")
            raise Exception("no rectagnle prompt!")
        
        #通过rectangle矩形框获取tile位置
        input_data  =  rectangle_data
        point_x     = input_data['value']['x'] / 100
        point_y     = input_data['value']['y'] / 100
        box_width   = input_data['value']['width'] / 100 
        box_height  = input_data['value']['height'] / 100
        
        logger.debug(f"point_x:{point_x},point_y:{point_y},box_width:{box_width},box_height:{box_height}")
        #左上角tile瓦片位置
        layer_x_position = int(point_x * current_layer_width)  #通过layer层位置计算出框坐标点的width
        layer_y_position = int(point_y * current_layer_height) #通过layer层位置计算出框坐标点的height
        #计算出rectangle坐标点瓦片位置
        layer_x_min      = layer_x_position // tile_width
        layer_y_min      = layer_y_position // tile_height
        #计算出rectangle的宽、高位置
        layer_x_max      = int((point_x + box_width) * current_layer_width // tile_width)
        layer_y_max      = int((point_y + box_height) * current_layer_height // tile_height)

        slice_width_num  = layer_x_max - layer_x_min + 1
        slice_height_num = layer_y_max - layer_y_min + 1
        tile_size        = ( tile_width , tile_height )
        slice_size_num   = ( slice_width_num, slice_height_num )

        logger.debug(f"layer_x_min:{layer_x_min},layer_x_max:{layer_x_max}")
        logger.debug(f"layer_y_min:{layer_y_min},layer_y_max:{layer_y_max}")
        box_point_list = []
        for i in range(layer_y_min, layer_y_max + 1):
            for j in range(layer_x_min, layer_x_max + 1):
                box_point_list.append([j,i])
        
        logger.debug(f"box_point_list:{box_point_list}")
        #检查本地是否存在瓦片图:

        #图片下载、拼接
        #tile_image_url:瓦片图前缀
        tile_image_url = CONFIG.svs_tile_imageURL + image_filename

        #判断图片收集否存在(优化)
        slice_url, slice_width, slice_height = self.create_image(tile_size, slice_size_num, current_layer, 
                                      box_point_list, tile_image_url, image_filename)
        
        #修改tasks,contex的值,x,y,width,height都要改
        tasks[0]['data']['image']     = slice_url
        input_data['original_width']  = slice_width
        input_data['original_height'] = slice_height

        #对prompt中的坐标进行转换:
        #x,y,width,height位置变换:在拼接的slice中，输入点相对于参考点的坐标位置
        context['layer_cur_scale'] = layer_cur_scale
        for ctx in context['result']:
            point_x_ins = ctx['value']['x'] / 100
            point_y_ins = ctx['value']['y'] / 100
            ctx['value']['x'] = (point_x_ins * current_layer_width - layer_x_min * tile_width) / slice_width * 100
            ctx['value']['y'] = (point_y_ins * current_layer_height - layer_y_min * tile_height) / slice_height * 100

            logger.debug(f"ctx['type']:{ctx['type']}, x_relative:{ctx['value']['x']},y_relative:{ctx['value']['y']}")
                         
            if ctx['type'] == "rectanglelabels":
                box_width_ins  = ctx['value']['width'] / 100
                box_height_ins = ctx['value']['height'] / 100
                ctx['value']['width']   = box_width_ins * current_layer_width / slice_width * 100
                ctx['value']['height']  = box_height_ins * current_layer_height / slice_height * 100
                
                logger.debug(f"box_width_relative:{ctx['value']['width']}, box_height_relative:{ctx['value']['height']}")


    def tiff_handler(self, tasks:List[Dict], context: Optional[Dict] = None, **kwargs):
        """
        Description: tiff类型SAM实时识别
        """
        pass

    def get_layer_level(self, layerInfo, curScale):
        level = 0
        for item in range(len(layerInfo)):
            if curScale < layerInfo[item]['curScale']:
                level = item + 1
            else:
                level += 1
                break
        return level

if __name__ == "__main__":
    c = InMemoryLRUDictCache(2)
    c.put(1, 1)
    c.put(2,2)
    c.put("test",{"a":1})
    print(c.get("test"))
    logger.debug(c.cache)