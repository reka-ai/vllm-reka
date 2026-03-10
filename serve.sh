#!/bin/bash

set -euo pipefail

MODEL_PATH=${1:-}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
GPU_MEM=${GPU_MEM:-0.95}
MAX_LEN=${MAX_LEN:-16384}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-20000}
MAX_IMAGES=${MAX_IMAGES:-6}
MAX_VIDEOS=${MAX_VIDEOS:-3}
TP_SIZE=${TP_SIZE:-1}
DTYPE=${DTYPE:-"bfloat16"}
VIDEO_SAMPLING=${VIDEO_SAMPLING:-"chunk"}
VIDEO_NUM_FRAMES=${VIDEO_NUM_FRAMES:-6}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"reka-edge-2603"}
QUANTIZATION=${QUANTIZATION:-"bitsandbytes"}

if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 <model-path> [additional vllm serve args]"
    echo ""
    echo "Example:"
    echo "  HOST=0.0.0.0 PORT=8000 $0 /models/reka-edge-2603 --max-num-seqs 32"
    exit 1
fi

shift

echo "Starting vLLM OpenAI API Server..."
echo "Model Path: $MODEL_PATH"
echo "Served Model Name: $SERVED_MODEL_NAME"
echo "Host: $HOST"
echo "Port: $PORT"
echo "GPU Memory: $GPU_MEM"
echo "Max Model Length: $MAX_LEN"
echo "Max Batched Tokens: $MAX_BATCH_TOKENS"
echo "Max Images per Prompt: $MAX_IMAGES"
echo "Max Videos per Prompt: $MAX_VIDEOS"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Data Type: $DTYPE"
echo "Video Sampling: $VIDEO_SAMPLING"
echo ""

export USE_IMAGE_PATCHING=${USE_IMAGE_PATCHING:-1}
export VLLM_VIDEO_LOADER_BACKEND=${VLLM_VIDEO_LOADER_BACKEND:-"yasa"}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_FLASH_ATTN_VERSION=${VLLM_FLASH_ATTN_VERSION:-3}
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-300}

if [[ -n "${VLLM_TORCH_PROFILER_DIR:-}" ]]; then
    export VLLM_TORCH_PROFILER_DIR
fi

vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tokenizer-mode "yasa" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --max-num-batched-tokens "$MAX_BATCH_TOKENS" \
    --limit-mm-per-prompt "{\"image\": $MAX_IMAGES, \"video\": $MAX_VIDEOS}" \
    --media-io-kwargs "{\"video\": {\"num_frames\": $VIDEO_NUM_FRAMES, \"sampling\": \"${VIDEO_SAMPLING}\"}}" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype "$DTYPE" \
    --chat-template-content-format openai \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --quantization "$QUANTIZATION" \
    "$@"
