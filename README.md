# Pipeline for Prompt Hallucination Reductions

This repository implements a modular pipeline for reducing prompt hallucinations in LLM answers using composable stages such as baseline ask, prompt optimization, chain-of-verification, self-correction, and external judgment. The pipeline is model-agnostic and supports ScaleDown prompt compression. It is designed to be easy to extend, debug, and evaluate.


## Features

- **Composable Stages**: Baseline, APO (prompt rewrite), CoVe (Chain-of-Verification), Self-correct, Judge, and more.
- **Early Exit Policies**: Oracle or LLM-based gates for efficiency.
- **Model Agnostic**: Easily plug in different LLM backends or adapters.
- **Prompt Compression**: Built-in support for ScaleDown prompt compression.
- **Debugging**: Verbose debug output and prompt/answer diffs.
- **Minimal Example Dataset**: For quick testing and demonstration.

- **Token Usage & Cost Tracking**: Track prompt/completion tokens and costs per stage
- **Budget Validation**: Set spending limits with automatic pipeline stopping
- **Quality Thresholds**: Early exit based on confidence scores (0-1)
- **Cost-Benefit Analysis**: Marginal accuracy gain per cost analysis
- **Optimization Recommendations**: AI-powered suggestions for pipeline improvement
- **Real-time Dashboard**: Streamlit-based monitoring interface
- **Publication-quality Visualizations**: Matplotlib/seaborn plots for analysis



## Getting Started

### Requirements

- Python 3.8+
- Access to [ScaleDown](https://scaledown.xyz/) API (or compatible endpoint).

Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables

Set these variables in your environment before running:

| Variable              | Description                                              | Example                                 |
|-----------------------|----------------------------------------------------------|-----------------------------------------|
| `SCALEDOWN_API_KEY`   | Your ScaleDown API key                                   | `sk-xxx...`                             |
| `SCALEDOWN_CHAT_URL`  | ScaleDown endpoint for generation/compression            | `https://api.scaledown.xyz/generate`    |
| `LLM_API_KEY`         | Your LLM API key (After you create a LLM plugin)         | `sk-xxx...`                             |
| `SD_MODEL_TARGET`     | Model name for main answer model (default: gemini/gemini-pro) | `gemini-2.5-flash`                     |
| `SD_MODEL_HELPER`     | Model name for helper (APO) model                        | `gemini-2.5-flash`                     |
| `SD_MODEL_JUDGE`      | Model name for judge model                               | `gemini-2.5-flash`                     |
| `SD_RATE_TARGET`      | Compression rate for target model (float, default: 0.7)  | `0.7`                                   |
| `SD_RATE_HELPER`      | Compression rate for helper model (float, default: 0.6)  | `0.6`                                   |
| `SD_RATE_JUDGE`       | Compression rate for judge model (float, default: 0.0)   | `0.0`                                   |
| `PIPE_DEBUG`          | Enable debug output (`1` for True, else False)           | `1`                                     |
| `PIPE_DEBUG_MAXLEN`   | Max length for debug output snippets (default: 220)      | `220`                                   |
| `PIPE_DEBUG_HTTP`     | Print HTTP payloads (`1` for True, else False)           | `1`                                     |
| `MAX_BUDGET_USD`      | Maximum budget in USD (default: 10.0)                   | `10.0`                                  |
| `ENABLE_TRACKING`     | Enable usage tracking (`1` for True, else False)        | `1`                                     |
| `TRACKING_DB_PATH`    | Database path for tracking (default: usage_tracking.db) | `usage_tracking.db`                      |
| `COVE_EXIT_CONFIDENCE`| Confidence threshold for CoVe stage (default: 0.9)      | `0.9`                                   |

You can set these in your shell like so:
```bash
export SCALEDOWN_API_KEY="sk-xxx..."
export SCALEDOWN_CHAT_URL="https://api.scaledown.xyz/generate"
export PIPE_DEBUG=1
```

### Running

Run the pipeline demo:
```bash
python main.py
```

This will process the sample dataset from `dataset.py` and print results, including per-stage outputs, diffs, token usage, and final answers.

### Run with Dashboard
```bash
# Terminal 1: Run pipeline
python main.py

# Terminal 2: Start dashboard
python run_dashboard.py
# Then open http://localhost:8501 in your browser
```

### Analyze Results
```bash
# Analyze a specific trace file
python analyze_results.py runs/trace_20231201_120000.jsonl

# Generate plots only
python analyze_results.py runs/trace_20231201_120000.jsonl --output-dir my_plots

# Analysis without plots
python analyze_results.py runs/trace_20231201_120000.jsonl --no-plots
```
---

**Contact:** For ScaleDown access or model endpoints, visit [scaledown.xyz](https://scaledown.xyz/).
