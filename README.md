[![Join our Discord](https://img.shields.io/badge/Discord-join%20now-blue?logo=discord)](https://link.reka.ai/discord)  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-_oItsMineZ)](https://huggingface.co/RekaAI/reka-edge-2603)

# vllm-reka

This plugin serves [Reka Edge](https://huggingface.co/RekaAI/reka-edge-2603) — a 7B multimodal model with frontier-class image understanding, video analysis, object detection, and tool use — via vLLM.

It registers model architectures, a custom tokenizer, and HuggingFace configs so that vLLM can load and serve Reka checkpoints out of the box.

## Quickstart

```bash
# 1. Install the plugin
uv sync

# 2. Download model weights (~14 GB)
pip install huggingface_hub
hf download RekaAI/reka-edge-2603 --local-dir ./models/reka-edge-2603

# 3. Start the server
bash ./serve.sh ./models/reka-edge-2603

# 4. Query it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"reka-edge-2603","messages":[{"role":"user","content":"Hello!"}]}'
```

## Requirements

- **GPU**: NVIDIA GPU, ideally with ≥24 GB VRAM. This has been tested to work on GTX 3090 GPUs with 40-50 tokens/s.
- **OS**: Linux with CUDA. macOS is not supported for serving.
- **Python**: 3.10 ≥ x > 3.14
- **vLLM**: 0.15.x (0.15.0 ≥ x > 0.16.0)

## Supported Models

| Model | Architecture | Vision Encoder | Description |
|---|---|---|---|
| **Reka Edge** | `Yasa2ForConditionalGeneration` | ConvNextV2 | 7B multimodal model (image + video) |

## Installation

Recommended (reproducible, uses `uv.lock`):

```bash
uv sync
```

Fallback with pip:

```bash
pip install -e .
```

Or with Poetry:

```bash
poetry install
```

The plugin registers itself via the `vllm.general_plugins` entry point — vLLM discovers it automatically once installed.

## Serving

### `serve.sh` (recommended)

Use `serve.sh` as the default entrypoint. It applies the plugin-specific defaults that this repo is tested with.

```bash
bash ./serve.sh <model-path>
```

Example with explicit host/port:

```bash
HOST=0.0.0.0 PORT=8000 bash ./serve.sh ./models/reka-edge-2603
```

You can also pass through additional `vllm serve` flags:

```bash
bash ./serve.sh ./models/reka-edge-2603 --max-num-seqs 32
```

### `serve.sh` configuration

Common environment variables:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | API port |
| `SERVED_MODEL_NAME` | `reka-edge-2603` | Model name exposed to OpenAI-compatible clients |
| `GPU_MEM` | `0.95` | `--gpu-memory-utilization` |
| `MAX_LEN` | `16384` | `--max-model-len` |
| `MAX_BATCH_TOKENS` | `20000` | `--max-num-batched-tokens` |
| `MAX_IMAGES` | `6` | Per-prompt image cap |
| `MAX_VIDEOS` | `3` | Per-prompt video cap |
| `VIDEO_NUM_FRAMES` | `6` | Frames sampled per video. Higher values improve temporal understanding but increase latency and memory usage. |
| `VIDEO_SAMPLING` | `chunk` | Video sampling strategy |
| `TP_SIZE` | `1` | Tensor parallel size |
| `DTYPE` | `bfloat16` | vLLM dtype |
| `QUANTIZATION` | `bitsandbytes` | Quantization backend (see [Quantization](#quantization)) |

Optional runtime env vars:

- `VLLM_TORCH_PROFILER_DIR` (only exported when set)
- `USE_IMAGE_PATCHING` (default `1`)
- `VLLM_VIDEO_LOADER_BACKEND` (default `yasa`)
- `VLLM_USE_V1` (default `1`)
- `VLLM_FLASH_ATTN_VERSION` (default `3`)
- `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` (default `300`)

### Quantization

The server defaults to 4-bit `bitsandbytes` quantization, which reduces VRAM usage enough to run on consumer GPUs (e.g., RTX 4090 with 24 GB). To run at full precision instead:

```bash
QUANTIZATION="" bash ./serve.sh ./models/reka-edge-2603
```

Full precision requires more VRAM (~14 GB in bfloat16) but avoids any quantization-related quality loss.

### Advanced: direct `vllm serve`

Prefer `serve.sh` unless you need full manual control. Minimal direct command:

```bash
vllm serve <model-path> \
  --tokenizer-mode yasa \
  --chat-template-content-format openai \
  --trust-remote-code
```

## Examples

Once the server is running, it exposes an OpenAI-compatible API at `http://localhost:8000` (or your configured `PORT`).

### Text

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reka-edge-2603",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Image understanding

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reka-edge-2603",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}},
        {"type": "text", "text": "Describe this image in detail."}
      ]
    }]
  }'
```

### Video analysis

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reka-edge-2603",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}},
        {"type": "text", "text": "Summarize what happens in this video."}
      ]
    }]
  }'
```

### Object detection

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reka-edge-2603",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}},
        {"type": "text", "text": "Detect: eye, ear"}
      ]
    }]
  }'
```

### Tool use / function calling

`serve.sh` enables tool use by default (`--enable-auto-tool-choice --tool-call-parser hermes`). Pass tools via the standard OpenAI `tools` parameter:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reka-edge-2603",
    "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

The model will return a `tool_calls` response when it decides to invoke a function.

### Python client

The server is compatible with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Text query
response = client.chat.completions.create(
    model="reka-edge-2603",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Multimodal query
response = client.chat.completions.create(
    model="reka-edge-2603",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}},
            {"type": "text", "text": "What's in this image?"},
        ],
    }],
)
print(response.choices[0].message.content)
```

## Monkey patches
This plugin contains several monkey patches because we originally developed our model on an older version of vLLM. In the long-term, we plan to open upstream PRs on the vLLM repo so that we can remove them.

Monkey patches were implemented for:
- Whitespace stripping introduced by our tiktoken-based tokenizer (`_patch_detokenizer_whitespace_stripping`)
- Add support for `YasaTokenizer` to construct `TokenizerInfo` with byte keys instead of just strings (`_patch_xgrammar_backend`)

## Troubleshooting

**`bitsandbytes` not found**

```
pip install bitsandbytes
```

Or disable quantization: `QUANTIZATION="" bash ./serve.sh ./models/reka-edge-2603`

**Out of memory (OOM)**

Lower GPU memory utilization or max sequence length:

```bash
GPU_MEM=0.85 MAX_LEN=8192 bash ./serve.sh ./models/reka-edge-2603
```

Or enable quantization (on by default): `QUANTIZATION=bitsandbytes bash ./serve.sh ./models/reka-edge-2603`

**"Model not found" or config errors**

Ensure the download path matches the path passed to `serve.sh`. The directory should contain `config.json`, `tokenizer.json`, and model weight files (`.safetensors`).

**Slow first request**

The first request after startup is slower because vLLM compiles CUDA graphs and warms up the KV cache. Subsequent requests will be significantly faster.

## Versioning
This plugin currently targets vLLM `0.15.x` (`>=0.15.0,<0.16.0`) and is tested on Linux. Since vLLM moves fast, we aim to update this plugin promptly as APIs evolve. If you are pinned to an older vLLM line, install a matching older plugin release.

## Dependencies

- `tiktoken` — tokenization
- `regex` — pattern matching for the tokenizer
- `opencv-python` — video frame extraction
- `numpy`, `Pillow` — image preprocessing

Requires Python ≥3.10, <3.14.

## License

Apache 2.0 — see [LICENSE](LICENSE).
