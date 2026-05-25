#!/bin/bash
shell_path=`cd $(dirname $0);pwd`
dir_path=`dirname $shell_path`

cd $dir_path

# 先保存调用方传入的环境变量（优先级高于 .env），再 source .env 作为默认值
_PRE_SERVER_PORT=${SERVER_PORT:-}
_PRE_WORKERS=${GUNICORN_WORKERS:-}
_PRE_THREADS=${GUNICORN_THREADS:-}
_PRE_TIMEOUT=${GUNICORN_TIMEOUT:-}
_PRE_POOL=${PREDICTOR_POOL_SIZE:-}

# .env 可选：容器部署时建议通过 --env-file 注入，本地开发时使用仓库根目录的 .env
if [ -f ./.env ]; then
    source ./.env
fi

# 恢复调用方传入的值（非空则覆盖 .env 中的同名变量）
[ -n "$_PRE_SERVER_PORT" ] && SERVER_PORT=$_PRE_SERVER_PORT
[ -n "$_PRE_WORKERS"     ] && GUNICORN_WORKERS=$_PRE_WORKERS
[ -n "$_PRE_THREADS"     ] && GUNICORN_THREADS=$_PRE_THREADS
[ -n "$_PRE_TIMEOUT"     ] && GUNICORN_TIMEOUT=$_PRE_TIMEOUT
[ -n "$_PRE_POOL"        ] && PREDICTOR_POOL_SIZE=$_PRE_POOL

# ── Gunicorn 配置 ──────────────────────────────────────────────────────────
# GUNICORN_WORKERS : 进程数，通常与 GPU 数量保持一致，默认 1
#   - 多进程会各自独立加载模型，显存消耗 = workers × 单模型显存
# GUNICORN_THREADS : 每个进程的线程数，处理并发 I/O（下载/拼图），默认 4
#   - GPU 推理通过内部 Lock 串行，线程数增加不会多占显存
# GUNICORN_TIMEOUT : 请求超时秒数，WSI 推理耗时较长，默认 120s
# ──────────────────────────────────────────────────────────────────────────
WORKERS=${GUNICORN_WORKERS:-1}
THREADS=${GUNICORN_THREADS:-4}
TIMEOUT=${GUNICORN_TIMEOUT:-120}

export PYTHONPATH="$dir_path/src:$PYTHONPATH"
export ENV_PATH="$dir_path/.env"

exec gunicorn \
  --bind "0.0.0.0:${SERVER_PORT}" \
  --workers "$WORKERS" \
  --worker-class gthread \
  --threads "$THREADS" \
  --timeout "$TIMEOUT" \
  --log-level info \
  mdi_sam_server.wsgi:app