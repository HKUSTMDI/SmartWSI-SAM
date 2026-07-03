# WSI 阅览图上 SAM 分割标注技术文档

> 🌐 **语言**：中文 · [English](./wsi-sam-segmentation-en.md)　|　📚 **相关文档**：[项目主页 (README)](../README.md) · [接口文档 (API)](./api.md)

## 1. 概述

### 1.1 背景

Whole Slide Image (WSI，全切片图像) 是数字病理学中的核心数据形式。一张 WSI 图像通常包含数十亿像素，分辨率极高，无法直接加载到 GPU 内存中进行 SAM 推理。本系统实现了一套完整的 **"WSI 瓦片拼接 + SAM 实时分割"** 方案，使得标注人员可以在 WSI 阅览图上通过画框/打点的方式进行实时交互式分割标注。

### 1.2 系统能力

- **多格式支持**：SDPC、SVS、TIFF 三种主流 WSI 格式
- **双提示模式**：支持 `keypointlabels`(点标注) 和 `rectanglelabels`(矩形框标注)，可混合使用 positive/negative 点
- **多模型支持**：SAM、SAM2、MobileSAM、ONNX 四种推理后端
- **多 GPU 并行**：PredictorPool 支持跨 GPU 实例池，实现真正的并行推理
- **Label Studio 集成**：可直接作为 Label Studio ML Backend 使用

---

## 2. 系统架构

### 2.1 整体架构

下图展示了 WSI 实时 SAM 分割的完整框架流程，共分为四层：用户交互层、API 网关层、WSI 瓦片处理层、SAM 推理层。

<p align="center">
  <img src="../demo/Structure.png" alt="WSI Real-Time SAM Segmentation Framework" width="70%" />
</p>

文本版详细架构图如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                     Label Studio Frontend                        │
│  (用户画框/打点 → 发送 predict 请求 → 渲染返回的 mask)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP POST /api/predict
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Flask API Layer (api.py)                        │
│  - 路由 img_type → sdpc / svs / tiff / normal                    │
│  - 调用 wsiHandler 处理 WSI 瓦片                                  │
│  - 调用 SamMLBackend.predict() 执行 SAM 推理                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
   ┌──────────────┐ ┌────────────┐ ┌──────────────┐
   │  wsiHandler  │ │ wsiHandler │ │    normal    │
   │  .sdpc_convert│ │.svs_handler│ │  (直接推理)   │
   │  (SDPC瓦片)  │ │ (SVS瓦片)  │ │              │
   └──────┬───────┘ └─────┬──────┘ └──────┬───────┘
          │               │               │
          ▼               ▼               │
   ┌──────────────────────────────────────┐│
   │     create_image() 瓦片拼接           ││
   │  - 下载 N 个 tile (asyncio 并发)      ││
   │  - 拼接为一张 slice 图                ││
   │  - 本地缓存 (LOCAL_STORAGE)           ││
   └──────────────────┬───────────────────┘│
                      │                    │
                      ▼                    ▼
          ┌──────────────────────────────────────┐
          │        SamMLBackend (model.py)        │
          │  - 解析 prompt (point/box)            │
          │  - 调用 PredictorPool.acquire()       │
          │  - 转换 mask → RLE 格式               │
          │  - 返回 Label Studio 格式结果          │
          └──────────────────┬───────────────────┘
                             │
                             ▼
          ┌──────────────────────────────────────┐
          │     PredictorPool (sam_predictor.py)  │
          │  - N 个 SAMPredictor 实例             │
          │  - Round-robin 分配 GPU 设备          │
          │  - 队列获取/归还实例                   │
          └──────────────────┬───────────────────┘
                             │
                             ▼
          ┌──────────────────────────────────────┐
          │    SAMPredictor.predict_sam()         │
          │  - set_image() + embedding 缓存       │
          │  - predictor.predict(points, box)     │
          │  - findContours → bbox               │
          │  - 返回 mask ndarray                  │
          └──────────────────────────────────────┘
```

### 2.2 核心文件说明

| 文件 | 职责 |
|------|------|
| [api.py](../src/mdi_sam_server/label_studio_ml_mdi/api.py) | Flask 路由层，分发不同 `img_type` 到对应处理逻辑 |
| [utils.py](../src/mdi_sam_server/label_studio_ml_mdi/utils.py) | WSI 瓦片处理、图片下载拼接、坐标转换、缓存清理 |
| [model.py (sam_backend)](../src/mdi_sam_server/sam_backend/model.py) | SAM 推理业务层，prompt 解析 → 模型推理 → RLE 结果 |
| [sam_predictor.py](../src/mdi_sam_server/sam_backend/sam_predictor.py) | SAM 模型封装，支持 SAM/SAM2/MobileSAM/ONNX |
| [settings.py](../src/mdi_sam_server/label_studio_ml_mdi/conf/settings.py) | 环境配置：瓦片服务前缀、缓存路径、清理参数 |

---

## 3. 核心工作流程

### 3.1 端到端时序

```
用户操作                Label Studio Frontend          MDI SAM Server                  WSI Tile Service
────────                ────────────────────          ──────────────                  ────────────────
  │                           │                            │                               │
  │  1. 加载WSI阅览图          │                            │                               │
  │  2. 画矩形框/打点          │                            │                               │
  ├──────────────────────────►│                            │                               │
  │                           │  3. POST /api/predict      │                               │
  │                           │  {img_type:"svs",          │                               │
  │                           │   context:{cur_scale,      │                               │
  │                           │   result:[...]}}           │                               │
  │                           ├───────────────────────────►│                               │
  │                           │                            │  4. api._predict()            │
  │                           │                            │     ├─ 解析 context            │
  │                           │                            │     ├─ img_type="svs"         │
  │                           │                            │     └─ wsiHandler.svs_handler │
  │                           │                            │                               │
  │                           │                            │  5. 获取 WSI 元信息            │
  │                           │                            ├──────────────────────────────►│
  │                           │                            │◄──────────────────────────────┤
  │                           │                            │   layerSize, tileSize,        │
  │                           │                            │   sliceLayerInfo              │
  │                           │                            │                               │
  │                           │                            │  6. 计算瓦片位置               │
  │                           │                            │     rectangle坐标→tile索引     │
  │                           │                            │     layer_x_min~max           │
  │                           │                            │                               │
  │                           │                            │  7. 并发下载瓦片               │
  │                           │                            ├──────────────────────────────►│
  │                           │                            │  GET /tile/layer/x/y (N个请求) │
  │                           │                            │◄──────────────────────────────┤
  │                           │                            │  tile JPEG images              │
  │                           │                            │                               │
  │                           │                            │  8. 瓦片拼接                   │
  │                           │                            │     Image.new().paste()        │
  │                           │                            │     保存为本地 JPEG             │
  │                           │                            │                               │
  │                           │                            │  9. SAM 推理                   │
  │                           │                            │     predictor.set_image()      │
  │                           │                            │     predictor.predict()        │
  │                           │                            │     返回 mask ndarray          │
  │                           │                            │                               │
  │                           │                            │  10. mask → RLE 转换           │
  │                           │                            │      brush.mask2rle()          │
  │                           │◄───────────────────────────┤                               │
  │                           │  11. predictions            │                               │
  │                           │  {results:[{type:"brush-   │                               │
  │                           │   labels", value:{rle,     │                               │
  │                           │   bbox, brushlabels}}]}    │                               │
  │◄──────────────────────────┤                            │                               │
  │  12. 渲染 mask 叠加层      │                            │                               │
```

---

## 4. WSI 图片处理详解

### 4.1 WSI 的层级 (Layer/Pyramid) 结构

WSI 图像通常以**金字塔结构**存储，包含多个分辨率层级：

- **Layer 0**（最高分辨率）：原始扫描层，例如 100000×50000 像素
- **Layer 1, 2, ...**：逐级下采样层，每层尺寸约为上一层的 1/4
- **阅览图层**：用户当前浏览的分辨率级别，由 `cur_scale` 参数标识

每个 Layer 被进一步切分为固定大小的 **Tile（瓦片）**，例如 256×256 或 512×512 像素，通过 REST API 按需获取。

### 4.2 img_type 路由分发 ([api.py:39-76](src/mdi_sam_server/label_studio_ml_mdi/api.py#L39-L76))

```python
@_server.route('/api/predict', methods=['POST'])
def _predict():
    img_type = data.get('img_type', 'normal')

    if img_type == "normal":
        # 普通图片：直接走 SAM 推理
        predictions = model.predict(tasks, context=context, **params)

    elif img_type == "sdpc":
        # SDPC 格式：先做瓦片拼接，再推理
        wsi_handler.sdpc_convert(tasks, context=context, **params)
        predictions = model.predict(tasks, context=context, **params)

    elif img_type == "svs" or img_type == "tiff":
        # SVS/TIFF 格式：先做瓦片拼接，再推理
        wsi_handler.svs_handler(tasks, context=context, **params)
        predictions = model.predict(tasks, context=context, **params)
```

**关键设计**：WSI 类型的处理是**先转换 tasks 中的数据**（用拼接后的 slice 图 URL 替换原始 WSI URL），然后走**与普通图片完全相同的 SAM 推理流程**。这种设计使得 WSI 处理和 SAM 推理完全解耦。

### 4.3 瓦片位置计算

#### SDPC 格式 ([utils.py:348-366](src/mdi_sam_server/label_studio_ml_mdi/utils.py#L348-L366))

SDPC 使用**百分比坐标系统**（0-100），通过 `sliceNumX/sliceNumY` 计算瓦片位置：

```python
# 用户在阅览图上画的矩形框，所有坐标都是百分比 (0-100)
point_x     = input_data['value']['x'] / 100      # 百分比 → 小数
point_y     = input_data['value']['y'] / 100
box_width   = input_data['value']['width'] / 100
box_height  = input_data['value']['height'] / 100

# 通过当前 layer 的切片数量计算瓦片索引
current_layer_sliceX = current_layer_info['sliceNumX']
current_layer_sliceY = current_layer_info['sliceNumY']

layer_x_min = floor(point_x * current_layer_sliceX)
layer_y_min = floor(point_y * current_layer_sliceY)
layer_x_max = floor((point_x + box_width) * current_layer_sliceX)
layer_y_max = floor((point_y + box_height) * current_layer_sliceY)
```

#### SVS 格式 ([utils.py:449-462](src/mdi_sam_server/label_studio_ml_mdi/utils.py#L449-L462))

SVS 使用**像素坐标系统**，需要先换算为当前 layer 上的像素坐标，再除以 tile 大小得到瓦片索引：

```python
# 当前 layer 的宽高（像素）
current_layer_width  = current_layer_info['sliceWidth']
current_layer_height = current_layer_info['sliceHeight']

# rectangle 坐标点像素位置 = 百分比 * layer 像素宽度
layer_x_position = int(point_x * current_layer_width)
layer_y_position = int(point_y * current_layer_height)

# 瓦片索引 = 像素位置 ÷ 瓦片大小
layer_x_min = layer_x_position // tile_width
layer_y_min = layer_y_position // tile_height
layer_x_max = int((point_x + box_width) * current_layer_width // tile_width)
layer_y_max = int((point_y + box_height) * current_layer_height // tile_height)
```

### 4.4 Layer 级别选择 ([utils.py:513-521](src/mdi_sam_server/label_studio_ml_mdi/utils.py#L513-L521))

系统根据用户当前浏览的 `cur_scale` 自动匹配最合适的图层级别：

```python
def get_layer_level(self, layerInfo, curScale):
    level = 0
    for item in range(len(layerInfo)):
        if curScale < layerInfo[item]['curScale']:
            level = item + 1   # curScale 小于该层 → 继续向上找
        else:
            level += 1         # curScale >= 该层 → 使用当前层
            break
    return level
```

**选层逻辑**：选择 `curScale` 最接近但不低于该层 `curScale` 的 Layer。这样可以确保瓦片下载的分辨率不低于用户实际看到的分辨率。

---

## 5. 瓦片下载与图片拼接

下图完整展示了 WSI 瓦片处理的算法细节，涵盖三大部分：**输入空间**（金字塔层级选择与坐标→瓦片网格映射）、**下载拼接流水线**（URL 构造、异步并发下载、网格拼接、SHA-256 缓存判断）、**输出空间**（WSI 全局坐标 → Slice 局部坐标的反变换）。这对应本文档第 4~6 章的核心逻辑。

<p align="center">
  <img src="../demo/WSI_slice.png" alt="WSI Tile Processing Algorithm Detail" width="100%" />
</p>

### 5.1 异步并发下载 ([utils.py:136-167](src/mdi_sam_server/label_studio_ml_mdi/utils.py#L136-L167))

系统使用 `aiohttp` + `asyncio` 实现瓦片的异步并发下载，支持失败重试（最多 3 次）：

```python
async def download_image(session, image):
    url = image['tile_url']
    max_retry = 3
    retry_count = 0

    while retry_count < max_retry:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    image_ins = Image.open(BytesIO(image_data))
                    image['content'] = image_ins
                    image['width'] = image_ins.size[0]
                    image['height'] = image_ins.size[1]
                else:
                    raise Exception("Failed to download image")
            break
        except Exception as e:
            retry_count += 1
            if retry_count < max_retry:
                await asyncio.sleep(2)  # 延时重试
                continue
            else:
                raise e
```

### 5.2 瓦片拼接 ([utils.py:221-300](src/mdi_sam_server/label_studio_ml_mdi/utils.py#L221-L300))

`create_image()` 方法将多个瓦片按网格排列拼接为一张完整的 slice 图：

```python
# 计算拼接图的宽高
slice_width = 0
slice_height = 0
for item, image in enumerate(image_url_list):
    if item // slice_size_num[0] == 0:
        slice_width += image['width']      # 累加第一行的宽度
    if item % slice_size_num[0] == 0:
        slice_height += image['height']    # 累加第一列的高度

# 创建空白画布
slice_image = Image.new('RGB', (slice_width, slice_height), 'white')

# 按网格粘贴瓦片
for item, image in enumerate(image_url_list):
    slice_x = int(item % slice_size_num[0] * tile_size[0])   # 列索引 × 瓦片宽度
    slice_y = int(item // slice_size_num[0] * tile_size[1])  # 行索引 × 瓦片高度
    slice_image.paste(image["content"], (slice_x, slice_y))
```

### 5.3 拼接图缓存

拼接后的图片会以**哈希命名**保存到本地磁盘，下次相同请求直接命中缓存：

```python
# 文件名 = 原始名 + layer + 坐标hash
hash_name = hashlib.sha256(str(point_list).encode('utf-8')).hexdigest()
slice_filename = image_filename + '_' + str(layer) + '_' + hash_name + '.jpeg'
local_slice_filename = os.path.join(CONFIG.local_storage, slice_filename)

# 缓存命中：直接从文件读取尺寸，跳过所有网络 I/O
if os.path.exists(local_slice_filename):
    with Image.open(local_slice_filename) as cached_img:
        slice_width, slice_height = cached_img.size
    return url_slice_filename, slice_width, slice_height
```

---

## 6. 坐标转换详解

### 6.1 转换目的

用户在 WSI 阅览图上画框时，坐标是基于**整个 WSI 的百分比坐标**。但在瓦片拼接后，实际上传给 SAM 推理的是一张**裁剪后的局部 slice 图**。因此需要将 prompt 坐标从"WSI 全局百分比坐标"转换为"slice 图局部百分比坐标"。

### 6.2 SDPC 坐标转换 ([utils.py:381-398](src/mdi_sam_server/label_studio_ml_mdi/utils.py#L381-L398))

```python
# 关键公式
for ctx in context['result']:
    point_x_ins = ctx['value']['x'] / 100   # 输入：WSI 全局百分比(0-100) → 小数
    point_y_ins = ctx['value']['y'] / 100

    # 转换：slice 图上的相对百分比
    # (全局小数 × 该层总切片数 - 起始切片索引) / 拼接的切片数 × 100
    ctx['value']['x'] = (point_x_ins * current_layer_sliceX - layer_x_min) / slice_width_num * 100
    ctx['value']['y'] = (point_y_ins * current_layer_sliceY - layer_y_min) / slice_height_num * 100

    # 矩形框宽高同理
    if ctx['type'] == "rectanglelabels":
        box_width_ins = ctx['value']['width'] / 100
        ctx['value']['width']  = box_width_ins * current_layer_sliceX / slice_width_num * 100
        ctx['value']['height'] = box_height_ins * current_layer_sliceY / slice_height_num * 100
```

### 6.3 SVS 坐标转换 ([utils.py:488-504](src/mdi_sam_server/label_studio_ml_mdi/utils.py#L488-L504))

SVS 使用像素坐标系，转换公式略有不同：

```python
# SVS 坐标转换
ctx['value']['x'] = (point_x_ins * current_layer_width - layer_x_min * tile_width) / slice_width * 100
ctx['value']['y'] = (point_y_ins * current_layer_height - layer_y_min * tile_height) / slice_height * 100

# 矩形框
ctx['value']['width']  = box_width_ins * current_layer_width / slice_width * 100
ctx['value']['height'] = box_height_ins * current_layer_height / slice_height * 100
```

### 6.4 转换示意图

```
┌─────────────────────────────────────┐
│          WSI 全局 (100×100%)         │
│                                     │
│      ┌─────────────────┐            │
│      │  用户画的矩形框    │            │
│      │  x=67.65, y=37.3 │            │
│      │  w=0.2,   h=0.2  │            │
│      └────────┬────────┘            │
│               │                     │
│               │ 瓦片拼接 + 坐标转换    │
│               ▼                     │
│  ┌─────────────────────────────┐    │
│  │     Slice 图 (局部)          │    │
│  │                             │    │
│  │  ┌───────────────┐          │    │
│  │  │ 转换后的坐标    │          │    │
│  │  │ x=45.2, y=33.1 │          │    │
│  │  │ w=15.8, h=12.3 │          │    │
│  │  └───────────────┘          │    │
│  └─────────────────────────────┘    │
│                                     │
└─────────────────────────────────────┘
```

---

## 7. SAM 模型推理

### 7.1 Prompt 解析 ([model.py:19-68](src/mdi_sam_server/sam_backend/model.py#L19-L68))

SamMLBackend 从 context 中解析用户输入的两种 prompt：

```python
def predict(self, tasks, context=None, **kwargs):
    point_coords = []   # [[x1,y1], [x2,y2], ...]
    point_labels = []   # [1, 1, 0, ...]  1=positive, 0=negative
    input_box = None    # [x, y, x+w, y+h]

    for ctx in context['result']:
        x = ctx['value']['x'] * image_width / 100    # 百分比 → 像素坐标
        y = ctx['value']['y'] * image_height / 100

        if ctx_type == 'keypointlabels':
            point_coords.append([int(x), int(y)])
            point_labels.append(int(ctx['is_positive']))  # 正/负样本点

        elif ctx_type == 'rectanglelabels':
            box_width  = ctx['value']['width'] * image_width / 100
            box_height = ctx['value']['height'] * image_height / 100
            input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]
```

**两种 Prompt 可以混合使用**：例如画一个矩形框限定大范围，再打几个 keypoint 精确指示目标。

### 7.2 SAM 推理 ([sam_predictor.py:193-239](src/mdi_sam_server/sam_backend/sam_predictor.py#L193-L239))

```python
def predict_sam(self, img_path, point_coords=None, point_labels=None, input_box=None):
    # Step 1: 加载图片并计算 image embedding
    self.set_image(img_path, calculate_embeddings=False)

    # Step 2: 调用 SAM predictor
    masks, probs, logits = self.predictor.predict(
        point_coords=point_coords,    # 正负样本点
        point_labels=point_labels,    # 1/0 标签
        box=input_box,                # 矩形框 [x,y,x2,y2]
        multimask_output=False        # 单 mask 输出
    )

    # Step 3: 从 mask 计算外接矩形 (bbox)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours)

    return {
        'masks': [mask],
        'probs': [float(probs[0])],
        'bbox':  [x, y, w, h]
    }
```

### 7.3 Mask → RLE 格式 ([model.py:70-99](src/mdi_sam_server/sam_backend/model.py#L70-L99))

SAM 输出的 mask 是二值 numpy 数组，需要转换为 Label Studio 支持的 RLE (Run-Length Encoding) 格式：

```python
def get_results(self, masks, probs, width, height, bbox, label, cur_scale):
    results = []
    for mask, prob in zip(masks, probs):
        mask = mask * 255                    # 二值→255
        rle = brush.mask2rle(mask)           # numpy → RLE

        results.append({
            'original_width': width,
            'original_height': height,
            'layer_cur_scale': cur_scale,     # WSI 专用：记录当前层 scale
            'value': {
                'format': 'rle',
                'rle': rle,                   # RLE 编码
                'bbox': bbox,                 # 外接矩形
                'brushlabels': [label],
            },
            'score': prob,
            'type': 'brushlabels',
        })
    return [{'result': results, 'model_version': PREDICTOR_POOL.model_name}]
```

---

## 8. 缓存策略

### 8.1 多级缓存体系

| 缓存层级 | 实现 | 作用 |
|----------|------|------|
| WSI 元信息缓存 | `InMemoryLRUDictCache(50)` | 缓存 WSI 的 sliceLayerInfo，避免重复 HTTP 请求 |
| 瓦片拼接图缓存 | `LOCAL_STORAGE` 磁盘缓存 | 相同区域+相同 layer 的拼接图直接复用 |
| Image Embedding 缓存 | `InMemoryLRUDictCache(1)` | 同一图片连续多次 predict 时复用 embedding |
| Predictor 实例池 | `PredictorPool` | 预加载 N 个模型实例，避免请求串行等待 |

### 8.2 缓存自动清理 ([utils.py:523-580](src/mdi_sam_server/label_studio_ml_mdi/utils.py#L523-L580))

后台线程定期清理过期的瓦片拼接图缓存：

```python
def start_cache_cleanup_scheduler():
    interval_seconds = CONFIG.cache_clean_interval_hours * 3600

    def _run():
        while True:
            time.sleep(interval_seconds)
            cleanup_local_storage()   # 删除超过 cache_max_age_hours 小时未访问的文件

    t = threading.Thread(target=_run, daemon=True, name="cache-cleanup")
    t.start()
```

配置项：

```shell
CACHE_MAX_AGE_HOURS=24         # 文件保留时长（小时）
CACHE_CLEAN_INTERVAL_HOURS=1   # 清理检查间隔（小时）
```

---

## 9. 性能与并发

### 9.1 多实例并行推理 ([sam_predictor.py:304-358](src/mdi_sam_server/sam_backend/sam_predictor.py#L304-L358))

`PredictorPool` 是系统的核心性能组件：

```python
class PredictorPool:
    def __init__(self, model_choice, pool_size):
        for i in range(pool_size):
            if gpu_count > 0:
                device = f"cuda:{i % gpu_count}"  # Round-robin 分配 GPU
            else:
                device = "cpu"
            predictor = SAMPredictor(model_choice, device=device)
            self._pool.put(predictor)

    @contextmanager
    def acquire(self):
        predictor = self._pool.get()   # 取出空闲实例（阻塞等待）
        try:
            yield predictor
        finally:
            self._pool.put(predictor)  # 归还实例
```

**工作机制**：
- 每个 `SAMPredictor` 实例绑定一个 GPU 设备
- 请求通过 `acquire()` 从队列取出空闲实例
- 推理完成后自动归还
- 多 GPU 场景下可实现真正的跨设备并行推理

### 9.2 性能数据

| 场景 | 串行 | 并发 | 加速比 |
|------|------|------|--------|
| Cache miss（需下载瓦片） | 0.713s | 0.145s | **4.9×** |
| Cache hit（瓦片已缓存） | 0.074s | 0.061s | ~1.2× |

---

## 10. API 使用指南

### 10.1 WSI 分割标注请求示例

#### SDPC 格式

```json
{
    "tasks": [{
        "data": {
            "image": "http://mdi.hkust-gz.edu.cn/wsi/sdpc/api/sliceInfo/sdpc/20211025_065925_0%238_11"
        }
    }],
    "task_id": "1",
    "img_type": "sdpc",
    "params": {
        "context": {
            "cur_scale": 1.1,
            "result": [
                {
                    "original_width": 3840,
                    "original_height": 2160,
                    "image_rotation": 0,
                    "value": {
                        "x": 67.65,
                        "y": 37.3,
                        "width": 0.2,
                        "height": 0.2,
                        "rectanglelabels": ["Tumor"]
                    },
                    "type": "rectanglelabels",
                    "origin": "manual"
                },
                {
                    "original_width": 3840,
                    "original_height": 2160,
                    "image_rotation": 0,
                    "value": {
                        "x": 67.72,
                        "y": 37.36,
                        "width": 0.2,
                        "keypointlabels": ["Tumor"]
                    },
                    "is_positive": true,
                    "type": "keypointlabels",
                    "origin": "manual"
                }
            ]
        }
    }
}
```

#### SVS 格式

```json
{
    "tasks": [{
        "data": {
            "image": "https://your-host/wsi/metaservice/api/sliceInfo/openslide/sample-001"
        }
    }],
    "task_id": "2",
    "img_type": "svs",
    "params": {
        "context": {
            "cur_scale": 0.03,
            "result": [
                {
                    "original_width": 1920,
                    "original_height": 1080,
                    "image_rotation": 0,
                    "value": {
                        "x": 50.0,
                        "y": 50.0,
                        "width": 5.0,
                        "height": 5.0,
                        "rectanglelabels": ["Cell"]
                    },
                    "type": "rectanglelabels",
                    "origin": "manual"
                }
            ]
        }
    }
}
```

### 10.2 响应格式

```json
{
    "results": [{
        "model_version": "SAM2:../models/sam2_hiera_base_plus.pt:cuda:0",
        "result": [{
            "id": "33ce",
            "original_width": 672,
            "original_height": 672,
            "image_rotation": 0,
            "layer_cur_scale": 1.0,
            "readonly": false,
            "score": 0.7698,
            "type": "brushlabels",
            "value": {
                "format": "rle",
                "rle": [12, 5, 8, 3, ...],
                "bbox": [400, 257, 51, 78],
                "brushlabels": ["Tumor"]
            }
        }]
    }]
}
```

### 10.3 重要参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `img_type` | string | **WSI 必传**。可选值: `sdpc` / `svs` / `tiff` / `normal` |
| `context.cur_scale` | float | **WSI 必传**。当前阅览图的 scale，用于选择正确的 WSI 层级 |
| `context.result[].type` | string | `keypointlabels`（点）或 `rectanglelabels`（矩形框） |
| `context.result[].is_positive` | bool | 仅 keypoint 模式。`true`=正样本，`false`=负样本 |
| `context.result[].value.x/y` | float | 0-100 的**百分比**坐标 |
| `context.result[].value.width/height` | float | 矩形框的百分比宽高 |
| `task_id` | string | 任务 ID，用于区分不同的标注任务 |

---

## 11. 环境配置

### 11.1 WSI 相关配置项

```shell
# SDPC 格式瓦片服务
SDPC_TILE_PREFIX=https://your-host/wsi/sdpc/api/sliceInfo/sdpc/
SDPC_TILE_IMAGEURL=https://your-host/wsi/sdpc/api/tile/sdpc/

# SVS/TIFF 格式瓦片服务
SVS_TILE_PREFIX=https://your-host/wsi/metaservice/api/sliceInfo/openslide/
SVS_TILE_IMAGEURL=https://your-host/wsi/metaservice/api/tile/openslide/

# 瓦片拼接图本地缓存目录
LOCAL_STORAGE=/home/mdi/.cache/label-studio/

# 缓存自动清理
CACHE_MAX_AGE_HOURS=24
CACHE_CLEAN_INTERVAL_HOURS=1
```

### 11.2 WSI 瓦片服务要求

系统依赖外部 WSI 瓦片服务提供以下 API：

1. **切片信息接口**：`GET ${PREFIX}/{image_id}`
   - 返回 `basisInfo`（tileWidth, tileHeight, layerSize）
   - 返回 `sliceLayerInfo` 或 `sliceInfo`（各层级元数据）

2. **瓦片图片接口**：`GET ${IMAGEURL}/{image_id}/{layer}/{x}/{y}`
   - 返回单个 tile 的 JPEG/PNG 图片

---

## 12. 关键设计要点

### 12.1 WSI 与普通图片的统一处理

核心设计理念：**WSI 的特殊处理仅限于"将 WSI URL + 坐标 → slice 图片 URL + 转换后的坐标"**，后续 SAM 推理完全复用普通图片的流程。这保证了：

- 代码耦合度低，WSI 处理逻辑（`utils.py`）和 SAM 推理逻辑（`model.py`）完全独立
- 新增 WSI 格式只需实现 `xxx_handler()` 方法
- SAM 模型升级不影响 WSI 处理

### 12.2 坐标系统的统一

所有 prompt 坐标均使用 **0-100 的百分比坐标**，与原始图片尺寸无关。`original_width/original_height` 参数用于在推理时还原为像素坐标。

### 12.3 缓存策略的分层设计

- **内存层**：高频热点数据（元信息、embedding），LRU 淘汰
- **磁盘层**：瓦片拼接图，基于内容的哈希命名 + 基于时间的过期清理
- **实例层**：PredictorPool 预加载模型实例，消除冷启动延迟
