# Pipeline for Prompt Hallucination Reductions

This repository implements a modular pipeline for reducing prompt hallucinations in LLM answers using composable stages such as baseline ask, prompt optimization, chain-of-verification, self-correction, and external judgment. The pipeline is model-agnostic and supports ScaleDown prompt compression. It is designed to be easy to extend, debug, and evaluate.

## Features

- **Composable Stages**: Baseline, APO (prompt rewrite), CoVe (Chain-of-Verification), Self-correct, Judge, and more.
- **Early Exit Policies**: Oracle or LLM-based gates for efficiency.
- **Model Agnostic**: Easily plug in different LLM backends or adapters.
- **Prompt Compression**: Built-in support for ScaleDown prompt compression.
- **Debugging**: Verbose debug output and prompt/answer diffs.
- **Minimal Example Dataset**: For quick testing and demonstration.

## Getting Started

### Requirements

- Python 3.8+
- `requests`
- Access to [ScaleDown](https://scaledown.xyz/) API (or compatible endpoint).

Install dependencies:
```bash
pip install requests
```

### Environment Variables

Create an .env by copying the contents of the .env.example

Set these variables in your environment before running:

| Variable              | Description                                              | Example                                 |
|-----------------------|----------------------------------------------------------|-----------------------------------------|
| `SCALEDOWN_API_KEY`   | Your ScaleDown API key                                   | `sk-xxx...`                             |
| `SCALEDOWN_CHAT_URL`  | ScaleDown endpoint for generation/compression            | `https://api.scaledown.xyz/generate`    |
| `LLM_API_KEY`         | Your LLM API key (After you create a LLM plugin)         | `sk-xxx...`                             |
| `SD_MODEL_TARGET`     | Model name for main answer model (default: gemini/gemini-pro) | `gemini/gemini-pro`                     |
| `SD_MODEL_HELPER`     | Model name for helper (APO) model                        | `gemini/gemini-pro`                     |
| `SD_MODEL_JUDGE`      | Model name for judge model                               | `gemini/gemini-pro`                     |
| `SD_RATE_TARGET`      | Compression rate for target model (float, default: 0.7)  | `0.7`                                   |
| `SD_RATE_HELPER`      | Compression rate for helper model (float, default: 0.6)  | `0.6`                                   |
| `SD_RATE_JUDGE`       | Compression rate for judge model (float, default: 0.0)   | `0.0`                                   |
| `PIPE_DEBUG`          | Enable debug output (`1` for True, else False)           | `1`                                     |
| `PIPE_DEBUG_MAXLEN`   | Max length for debug output snippets (default: 220)      | `220`                                   |
| `PIPE_DEBUG_HTTP`     | Print HTTP payloads (`1` for True, else False)           | `1`                                     |

### Running

Run the pipeline demo:
```bash
uv run main.py
```

This will process the sample dataset from `dataset.py` and print results, including per-stage outputs, diffs, token usage, and final answers.

### Custom Configuration

Pipeline stages and gate policies can be customized in `main.py` via the `config` dictionary:

```python
config = {
    "stages": [
        {"type":"baseline","id":"s0","model":"target"},
        {"type":"apo","id":"s1","helper":"helper","target":"target"},
        {"type":"cove","id":"s2","model":"target"},
        {"type":"self_correct","id":"s3","model":"target"},
        {"type":"judge","id":"s4","judge":"judge","exit_on_pass": True, "threshold": 0.8}
    ],
    "gate": {"mode": "judge", "judge": "judge", "threshold": 0.8, "template": ...},
    "token_diffs": True
}
```

See `main.py` or `depricated_pipeline.py.old` for further details and examples.

## File Overview

- `main.py` - Entry point and pipeline configuration.
- `pipeline.py` - Pipeline runner and configuration utilities.
- `stages.py` - Stage implementations and registry.
- `models.py` - Model adapters and wrappers (ScaleDown, etc.).
- `dataset.py` - Example question dataset for evaluation.
- `depricated_pipeline.py.old` - Older full-in-one script, for reference.
- `utils.py` - Some utilities

## Extending

- Add new stages by implementing the `Stage` protocol and registering via `@register_stage`.
- To add new models, implement the `Model` protocol.
- You can use or modify the provided prompt templates in `main.py` as needed.

## License

MIT License

---

**Contact:** For ScaleDown access or model endpoints, visit [scaledown.xyz](https://scaledown.xyz/).
