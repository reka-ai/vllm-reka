#!/bin/bash

set -e

# MODEL_PATH=${1:-"/app/anton/models/7b-stage3-v2-2026-01-23/"}
MODEL_PATH=${1:-"/home/donovan/vision/exps/yasa_edge_feb16/merge/feb16_1200_llm_only_from_29784_gs170_w0208"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-9000}
GPU_MEM=${GPU_MEM:-0.95}
MAX_LEN=${MAX_LEN:-8192}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-20000}
MAX_IMAGES=${MAX_IMAGES:-6}
MAX_VIDEOS=${MAX_VIDEOS:-3}
TP_SIZE=${TP_SIZE:-1}
DTYPE=${DTYPE:-"bfloat16"}
VIDEO_SAMPLING=${VIDEO_SAMPLING:-"chunk"}

# Shift to get additional arguments
shift 2>/dev/null || true

echo "Starting vLLM OpenAI API Server..."
echo "Model: $MODEL_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "GPU Memory: $GPU_MEM"
echo "Max Model Length: $MAX_LEN"
echo "Max Batched Tokens: $MAX_BATCH_TOKENS"
echo "Max Images per Prompt: $MAX_IMAGES"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Data Type: $DTYPE"
echo ""

# Set environment variable for torch profiler
#export VLLM_TORCH_PROFILER_DIR=/app/anton/traces/reka-spark-vl
#export VLLM_USE_V1=1
#export USE_IMAGE_PATCHING=1
#export VLLM_FLASH_ATTN_VERSION=3
#export VLLM_VIDEO_LOADER_BACKEND=yasa
#export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=300
#export YASA_EDGE_V2=$YASA_EDGE_V2

vllm serve "$MODEL_PATH" \
    --served-model-name yasa-edge-model \
    --host "$HOST" \
    --port "$PORT" \
    --tokenizer-mode "yasa" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --max-num-batched-tokens "$MAX_BATCH_TOKENS" \
    --limit-mm-per-prompt "{\"image\": $MAX_IMAGES, \"video\": $MAX_VIDEOS}" \
    --media-io-kwargs "{\"video\": {\"num_frames\": 6, \"sampling\": \"${VIDEO_SAMPLING}\"}}" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code
