# MDI SAM Server

**[Medical Data Intelligence Lab](https://mdi.hkust-gz.edu.cn/)**  ·  [Cheng ZHANG](https://zachczhang.github.io/)

![architecture](./docs/architecture.jpg)

MDI annotation platform real-time segmentation server powered by [SAM1](https://github.com/facebookresearch/segment-anything) & [SAM2](https://github.com/facebookresearch/segment-anything-2). Generate masks from point / box prompts on both standard images and Whole Slide Images (WSI). Can be used directly as a **Label Studio** ML backend.

**Features:**

- Real-time annotation with multi-point and rectangle prompts (positive / negative)
- Whole Slide Image (WSI) recognition — SVS / SDPC / TIFF formats
- Multi-threaded request handling via Gunicorn gthread (I/O-parallel, ~5x speedup on cache-miss)
- `PredictorPool` for multi-GPU parallel inference
- Auto cache cleanup for tile image storage

---

## Demo

### WSI Segmentation

<div>
<img src="./docs/demo1.gif" width="50%"/><img src="./docs/demo2.gif" width="50%"/>
</div>

### Point & Rectangle Mode

<p float="left">
  <img src="./docs/demo_point1.jpg" width="37%" />
  <img src="./docs/demo_point2.jpg" width="37%" />
  <br/>
  <img src="./docs/demo_rectangle1.jpg" width="37%" />
  <img src="./docs/demo_rectangle2.jpg" width="37%" />
</p>

### Supported Models

| # | Model | Notes |
|---|-------|-------|
| 1 | [Meta SAM](https://github.com/facebookresearch/segment-anything) | ViT-L/H |
| 2 | [Meta SAM2](https://github.com/facebookresearch/segment-anything-2) | **Recommended** |
| 3 | [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Lightweight |
| 4 | ONNX | CPU-friendly |

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (recommended; CPU mode also supported)
- NVIDIA GPU with ≥ 4 GB VRAM for SAM2-base+

### 1. Clone & Install

```shell
git clone https://github.com/HKUSTMDI/mdi-sam-server.git
cd mdi-sam-server

pip install -e .
```

### 2. Download Model Checkpoints

```shell
# Download all SAM2 checkpoints to ./models/
bash models/download_ckpts.sh

# Or manually place checkpoint files in ./models/:
#   sam2_hiera_tiny.pt     (~38 MB)
#   sam2_hiera_small.pt    (~184 MB)
#   sam2_hiera_base_plus.pt (~325 MB, default)
#   sam2_hiera_large.pt    (~898 MB)
#   sam_vit_l_0b3195.pth   (SAM1)
```

### 3. Configure Environment

Copy the example and fill in your values:

```shell
cp .env_example .env
```

Key variables in `.env`:

```shell
# Model selection
SAM_CHOICE=SAM2                  # SAM | SAM2 | MobileSAM | ONNX
SAM2_CHECKPOINT=/absolute/path/to/models/sam2_hiera_base_plus.pt
SAM2_CONFIG=sam2_hiera_b+.yaml

# Server
SERVER_PORT=9091

# WSI tile service endpoints (required for WSI segmentation)
SVS_TILE_PREFIX=https://your-host/wsi/metaservice/api/sliceInfo/openslide/
SVS_TILE_IMAGEURL=https://your-host/wsi/metaservice/api/tile/openslide/
SDPC_TILE_PREFIX=https://your-host/wsi/sdpc/api/sliceInfo/sdpc/
SDPC_TILE_IMAGEURL=https://your-host/wsi/sdpc/api/tile/sdpc/

# Tile image cache directory
LOCAL_STORAGE=/home/mdi/.cache/label-studio/

# Cache auto-cleanup
CACHE_MAX_AGE_HOURS=24           # Delete cached files older than N hours
CACHE_CLEAN_INTERVAL_HOURS=1     # Run cleanup every N hours

# Concurrency (see Performance section)
GUNICORN_WORKERS=1
GUNICORN_THREADS=4
GUNICORN_TIMEOUT=120
PREDICTOR_POOL_SIZE=1
```

### 4. Start Server

```shell
bash bin/start.sh
```

Health check:

```shell
curl http://localhost:9091/api/health
# {"code":200,"model_class":"SamMLBackend","msg":"ok"}
```

---

## Docker Deployment

### Build Image

```shell
docker build -t sam_server:latest .
```

> **Build cache optimization**: The Dockerfile uses layer splitting — dependencies are cached separately from source code. Re-building after code-only changes takes only a few seconds.
>
> If GitHub packages are slow to clone, the Dockerfile applies:
> - `git config http.version HTTP/1.1` — avoids HTTP/2 stream errors (curl 92)
> - GitHub proxy mirror for faster cloning in mainland China

### Run Container

```shell
docker run -d \
  --name mdi-sam \
  --restart always \
  --gpus all \
  -p 9091:9091 \
  -v $(pwd)/models:/app/models \
  -v /home/mdi/.cache/label-studio:/home/mdi/.cache/label-studio \
  --env-file .env \
  sam_server:latest
```

| Flag | Purpose |
|------|---------|
| `--gpus all` | Enable all GPUs |
| `-v $(pwd)/models:/app/models` | Mount model checkpoints (avoid re-downloading on restart) |
| `-v ...cache...` | Persist tile image cache across container restarts |
| `--env-file .env` | Load all configuration from `.env` |
| `--restart always` | Auto-restart on crash or host reboot |

### Override Variables at Runtime

```shell
# Run on a different port without editing .env
docker run -d --name mdi-sam --gpus all \
  -p 19091:19091 \
  -e SERVER_PORT=19091 \
  -e PREDICTOR_POOL_SIZE=2 \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  sam_server:latest
```

### Useful Docker Commands

```shell
# View logs
docker logs -f mdi-sam

# Stop / restart
docker stop mdi-sam
docker start mdi-sam

# Update restart policy on running container
docker update --restart=always mdi-sam

# Rebuild after code changes (deps layer is cached)
docker build -t sam_server:latest . && docker restart mdi-sam
```

---

## Performance & Concurrency

The server uses **Gunicorn gthread** worker mode, allowing multiple requests to be handled concurrently:

| Phase | Behavior | Config |
|-------|----------|--------|
| HTTP connection | Fully concurrent | `GUNICORN_THREADS` |
| WSI tile download | Fully parallel (I/O bound) | `GUNICORN_THREADS` |
| GPU inference | Serialized per predictor instance | `PREDICTOR_POOL_SIZE` |

**Benchmark results (3 concurrent requests, SVS image):**

```
Cache miss (tile download needed):
  Sequential: 0.713s  →  Concurrent: 0.145s  (4.9x speedup)

Cache hit (tiles already cached):
  Sequential: 0.074s  →  Concurrent: 0.061s  (~1.2x, near-instant)
```

### Multi-GPU Configuration

```shell
# 2 GPUs, 2 model instances (one per GPU) → true parallel inference
PREDICTOR_POOL_SIZE=2
GUNICORN_THREADS=8

# 4 GPUs
PREDICTOR_POOL_SIZE=4
GUNICORN_THREADS=16
```

VRAM requirements per instance:

| Model | VRAM |
|-------|------|
| SAM2-tiny | ~1 GB |
| SAM2-small | ~2 GB |
| SAM2-base+ | ~3-4 GB |
| SAM2-large | ~6-8 GB |

---

## API Reference

See [API Docs](./docs/api.md) for full details.

### `POST /api/predict`

Request body:

```json
{
  "tasks": [{"data": {"image": "<image_url>"}}],
  "params": {
    "context": {
      "result": [
        {
          "original_width": 1920,
          "original_height": 1080,
          "image_rotation": 0,
          "value": {
            "x": 50.0, "y": 50.0,
            "width": 5.0, "height": 5.0,
            "rectanglelabels": ["cell"]
          },
          "type": "rectanglelabels",
          "origin": "manual"
        }
      ],
      "cur_scale": 0.03
    }
  },
  "task_id": "1",
  "img_type": "svs"
}
```

`img_type`: `normal` | `svs` | `sdpc` | `tiff`

### `GET /api/health`

```json
{"code": 200, "model_class": "SamMLBackend", "msg": "ok"}
```

---

## Contact

If you find this project helpful, please give it a ⭐. For questions or issues, open an issue or email [czhangcn@connect.ust.hk](mailto:czhangcn@connect.ust.hk).

## License

Released under the [Apache-2.0 license](./LICENSE).

## Acknowledgement

We extend our thanks to the developers of [Label-Studio](https://github.com/HumanSignal/label-studio), [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/segment-anything-2), and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

## Citation

```bibtex
@misc{mdi-sam-server,
  year         = {2024},
  author       = {Cheng ZHANG},
  publisher    = {Github},
  journal      = {Github repository},
  title        = {{MDI SAM Server}: MDI machine learning SAM model service},
  url          = {https://github.com/HKUSTMDI/mdi-sam-server},
  organization = {HKUSTMDI}
}
```
