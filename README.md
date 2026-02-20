# vllm-reka

A [vLLM](https://github.com/vllm-project/vllm) plugin that adds support for serving Reka models. Registers model architectures, a custom tokenizer, and HuggingFace configs so that vLLM can load and serve Reka checkpoints out of the box.

## Supported Models

| Architecture | Config Type | Vision Encoder | Description |
|---|---|---|---|
| `YasaCausalLM` | `yasa_model` | — | Text-only LLM (LLaMA-style with SwiGLU, RoPE, GQA) |
| `YasaMMLMForConditionalGeneration` | `yasa_mmlm_model` | SigLIP | Multimodal model with image/video support |
| `Yasa2ForConditionalGeneration` | `yasa_edge_mmlm_model` | ConvNextV2 | Edge multimodal model (lighter vision encoder) |

## Installation

```bash
pip install -e .
```

Or with Poetry:

```bash
poetry install
```

The plugin registers itself via the `vllm.general_plugins` entry point — vLLM discovers it automatically once installed.

## Usage

### Prerequisites

Install vLLM first (see [vLLM installation docs](https://docs.vllm.ai/en/latest/getting_started/installation.html)):

```bash
pip install vllm
```

Then install this plugin:

```bash
pip install -e .
```

### Serving a text-only model

```bash
vllm serve <model-path> --tokenizer-mode yasa
```

### Serving a multimodal model

```bash
vllm serve <model-path> \
  --tokenizer-mode yasa
```

### Querying the server

Once running, the server exposes an OpenAI-compatible API at `http://localhost:8000`.

**Text completion:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-path>",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Multimodal (image):**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-path>",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }]
  }'
```

## Project Structure

```
vllm_reka/
├── __init__.py           # Plugin entry point — registers all components with vLLM
├── config.py             # HuggingFace PretrainedConfig classes (text, multimodal, edge)
├── tokenizer.py          # Tiktoken-based tokenizer with chat template and tool support
├── model.py              # YasaCausalLM — text-only transformer
├── multimodal_model.py   # SigLIP-based multimodal model
├── edge_model.py         # ConvNextV2-based edge multimodal model
└── multimodal_utils.py   # Image tiling, video frame sampling, preprocessing
```

## Dependencies

- `tiktoken` — tokenization
- `regex` — pattern matching for the tokenizer
- `opencv-python` — video frame extraction
- `numpy`, `Pillow` — image preprocessing

Requires Python ≥3.10, <3.14.

## License

Apache 2.0 — see [LICENSE](LICENSE).
