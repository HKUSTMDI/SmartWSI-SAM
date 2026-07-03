# 接口文档

> 📚 **相关文档**：[项目主页 (README)](../README.md) · WSI SAM 分割技术文档 [中文](./wsi-sam-segmentation.md) · [English](./wsi-sam-segmentation-en.md)

+ 说明:请求体采用json方式，请求头中包含token进行验证 
请求头:Content-Type:application/json;token:xxxx
## 接口列表:
接口名|描述
---|---
[predict](#predict)|sam预测接口
[preload](#preload )|SAM图片预加载接口，可提高后一步predict速度;<br>注意:<font color="red">wsi图片不能使用preload</font>
[health](#health)|健康检查

## predict
- 接口说明: 使用sam模型获取指定url图片位置的mask。如果url中的图片未经过preload **下载&加载图片进入模型**，届时将增加一定时耗。可先调用preload接口 **下载&加载图片进入模型**。
- method: **POST**
- URL: ```${prefix_url}/predict```
- body

参数名|类型|出现要求|描述
---|---|---|---
[tasks](#tasks参数)|json|是|指定的图片
task_id|string|是|任务id，用来区别不同task
[params](#params参数)|json|是|参数
model_version|string|否|sam模型类型,默认使用sam_vit_l
img_type|string|否|图片类型，不传默认为normal。可选:sdpc/svs/tiff/normal


+ tasks参数
```shell
{
    "tasks": [
        {
            'data': {
                'image': 'https: //s3.amazonaws.com/htx-pub/datasets/images/125245483_152578129892066_7843809718842085333_n.jpg'
            }
        }
    ]
}
```

+ params参数

```shell
#keypoint 点模式:
{   "login": null,
    "password": null,
    "context": {
        "cur_scale":0.23,#wsi类型图片专用
        "result": [
            {   
                "original_width": 3840,
                "original_height": 2160,
                "image_rotation": 0,
                "value": {
                    "x": 80.3072625698324,
                    "y": 43.67245657568238,
                    "width": 0.27932960893854747,
                    "keypointlabels": ["Banana"]
                },

                "is_positive": true,
                "type": "keypointlabels", //类型1:keypointlabels点标注,2:rectanglelabels方框标注
                "origin": "manual"
            },
            
        ]
    }
}

#rectangle 矩形模式:
{
    "login": null,
    "password": null,
    "context": {
        "result": [
            {
                "original_width": 3840,
                "original_height": 2160,
                "image_rotation": 0,
                "value": {
                    "x": 55,
                    "y": 44,
                    "width": 8,
                    "height": 10,
                    "rectanglelabels": [
                        "Banana"
                    ]
                },
                "type": "rectanglelabels",
                "origin": "manual"
            }
        ]
    }
}

```
+ request

Headers:None
Content-Type:application/json;token:test
<br>

+ response
```shell
{
    "results": [
        {
            "model_version": "SAM:../model/sam_vit_b_01ec64.pth:cuda",
            "result": [
                {
                    "id": "6386",
                    "image_rotation": 0,
                    "original_height": 2160,
                    "original_width": 3840,
                    "readonly": false,
                    "score": 0.7328885197639465,
                    "type": "brushlabels",
                    "value": {
                        "bbox": [
                            1501,
                            570,
                            642,
                            730
                        ],
                        "brushlabels": [
                            "Banana"
                        ],
                        "format": "rle",
                        "rle": [
                            ...
                            ]
                    }
                }
            ]
        }
    ]
}
```
### 演示
#### 1.keypoints prompt
此图传入了3个点，其中2个是positive，1个是negtive.
+ request
```shell
{
    "tasks": [
        {
            "data": {
                "image": "https://img.ibingli.cn/signature/CMWGTUhghiTnTpwd.jpg"
            }
        }
    ],
    "model_version": "mobile_sam",
    "task_id": 1,
    "params": {
        "login": null,
        "password": null,
        "context": {
            "result": [
                    {
                    "original_width": 3840,
                    "original_height": 2160,
                    "image_rotation": 0,
                    "value": {
                        "x": 50,
                        "y": 50,
                        "width": 0.3189792663476874,    
                        "keypointlabels": ["Banana"]
                    },
                    "is_positive": true,
                    "type": "keypointlabels",
                    "origin": "manual"
                },
                {
                    "original_width": 3840,
                    "original_height": 2160,
                    "image_rotation": 0,
                    "value": {
                        "x": 44,
                        "y": 50,
                        "width": 0.3189792663476874,    
                        "keypointlabels": ["Banana"]
                    },
                    "is_positive": true,
                    "type": "keypointlabels",
                    "origin": "manual"
                },
                {
                    "original_width": 3840,
                    "original_height": 2160,
                    "image_rotation": 0,
                    "value": {
                        "x": 50,
                        "y": 40,
                        "width": 0.3189792663476874,    
                        "keypointlabels": ["Banana"]
                    },
                    "is_positive": false,
                    "type": "keypointlabels",
                    "origin": "manual"
                }
            ]
        }
    }
}
```
+ response
  
```shell
{
    "results": [
        {
            "model_version": "SAM:../model/sam_vit_b_01ec64.pth:cuda",
            "result": [
                {
                    "id": "8c11",
                    "image_rotation": 0,
                    "original_height": 2160,
                    "original_width": 3840,
                    "readonly": false,
                    "score": 0.8481563925743103,
                    "type": "brushlabels",
                    "value": {
                        "bbox": [
                            1574,
                            952,
                            462,
                            198
                        ],
                        "brushlabels": [
                            "Banana"
                        ],
                        "format": "rle",
                        "rle": [
                            ...
                        ]
                    }
                }
            ]
        }
    ]
}
```

+ 效果:
<p float="left">
  <img src="../demo/demo_point2.jpg" width="37.25%" />
</p>

#### 2.rectangle prompt
传入一个矩形，进行分割识别。
+ request
```shell
{
    "tasks": [
        {
            "data": {
                "image": "https://img.ibingli.cn/signature/CMWGTUhghiTnTpwd.jpg"
            }
        }
    ],
    "model_version": "sam",
    "task_id": 1,
    "params": {
        "login": null,
        "password": null,
        "img_type": "sdpc", #wsi图片专用
        "context": {
            "result": [
                    {
                    "original_width": 3840,
                    "original_height": 2160,
                    "image_rotation": 0,
                    "value": {
                        "x": 55,
                        "y": 44,
                        "width": 8,
                        "height": 10,
                        "cur_scale":0.23,#wsi图片专用
                        "rectanglelabels": ["Banana"]
                    },
                    "type": "rectanglelabels",
                    "origin": "manual"
                }
            ]
        }
    }
}
```
+ response
```shell
{
    "results": [
        {
            "model_version": "SAM:../model/sam_vit_b_01ec64.pth:cuda",
            "result": [
                {
                    "id": "4a26",
                    "image_rotation": 0,
                    "original_height": 2160,
                    "original_width": 3840,
                    "readonly": false,
                    "score": 0.8841575980186462,
                    "type": "brushlabels",
                    "value": {
                        "bbox": [
                            2131,
                            968,
                            262,
                            162
                        ],
                        "brushlabels": [
                            "Banana"
                        ],
                        "format": "rle",
                        "rle": [
                            ...
                        ]
                    }
                }
            ]
        }
    ]
}        
```

效果图
<p float="left">
  <img src="../demo/demo_rectangle2.jpg" width="37.25%" float='left'/>
</p>

### 3.wsi predict方式:
针对wsi类型图片进行sam实时识别:
+ request
```shell
{
    "tasks": [
        {
            "data": {
                "image": "http://mdi.hkust-gz.edu.cn/wsi/sdpc/api/sliceInfo/sdpc/20211025_065925_0%238_11"
            }
        }
    ],
    "model_version": "sam",
    "task_id": 1,
    "img_type": "sdpc", #wsi预测需要传:sdpc/svs/tiff,
    "params": {
        "login": null,
        "password": null,
        "context": {
            "cur_scale":1.1, #wsi预测需要传
            "result": [
                #画框
                {
                    "original_width": 3840,
                    "original_height": 2160,
                    "image_rotation": 0,
                    "value": {
                        "x": 67.65,
                        "y": 37.3,
                        "width": 0.2,
                        "height": 0.2,
                        "rectanglelabels": ["Banana"]
                    },
                    "type": "rectanglelabels",
                    "origin": "manual"
                },
                #画positive点
                {
                    "original_width": 3840,
                    "original_height": 2160,
                    "image_rotation": 0,
                    "value": {
                        "x": 67.72,
                        "y": 37.36,
                        "width": 0.2,
                        "keypointlabels": ["Banana"]
                    },
                    "is_positive":true,
                    "type": "keypointlabels",
                    "origin": "manual"
                },
                #画positive点
                {
                    "original_width": 3840,
                    "original_height": 2160,
                    "image_rotation": 0,
                    "value": {
                        "x": 67.71,
                        "y": 37.45,
                        "width": 0.2,
                        "keypointlabels": ["Banana"]
                    },
                    "is_positive":true,
                    "type": "keypointlabels",
                    "origin": "manual"
                },
                #画negative点
                {
                    "original_width": 3840,
                    "original_height": 2160,
                    "image_rotation": 0,
                    "value": {
                        "x": 67.8,
                        "y": 37.41,
                        "width": 0.3,
                        "keypointlabels": ["Banana"]
                    },
                    "is_positive":false,
                    "type": "keypointlabels",
                    "origin": "manual"
                }

            ]
        }
    }
}
```
+ response
```
{
    "results": [
        {
            "model_version": "SAM:../models/sam_vit_l_0b3195.pth:cuda",
            "result": [
                {
                    "id": "33ce",
                    "image_rotation": 0,
                    "layer_cur_scale": 1, #layer的curScale
                    "original_height": 672, #拼接图的高
                    "original_width": 672,  #拼接图的宽
                    "readonly": false,
                    "score": 0.76984041929245,
                    "type": "brushlabels",
                    "value": {
                        "bbox": [
                            400,
                            257,
                            51,
                            78
                        ],
                        "brushlabels": [
                            "Banana"
                        ],
                        "format": "rle",
                        "rle": [
                            ...]
                    }
                }
            ]
        }
    ]
}
```
+ 效果图
<p float="left">
  <img src="../demo/wsgi_demo5.png" width="37.25%" float='left'/>
</p>

## preload 
- 接口说明:在使用sam模型进行自动标注前，前端加载过程中，可调用后端提前**下载&加载图片进入模型**。
- <font color='red'> 注意:wsi类型图片不支持 preload</font>
- method:**POST,GET**
- URL:```${prefix_url}/preload```
- body

参数名|类型|出现要求|描述
---|---|---|---
url|string|是|需要预下载图片的url
task_id|string|是|任务id

- request
Headers:
Content-Type:application/json;token:test
```
{
  "url":"xxxx",
  "task_id":"xxxx"
}
```
- response
```
{
    "code": 200,
    "msg": "ok"
}
```
## health
method: **GET**
URL:```${prefix_url}/health```

### request
Headers:
Content-Type:application/json;token:test
### response
```
{
    "code": 200,
    "model_class": "SamMLBackend",
    "msg": "ok"
}
``````