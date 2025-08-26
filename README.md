# Pipeline for Prompt Hallucination Reductions

This repository implements a modular pipeline for reducing prompt hallucinations in LLM answers using composable stages such as baseline ask, prompt optimization, chain-of-verification, self-correction, confidence checking, and external judgment. The pipeline supports multiple model backends (Ollama, Gemini, ScaleDown compression) with environment-variable based configuration.

## Features

- **Composable Stages**: Baseline, APO (prompt rewrite), CoVe (Chain-of-Verification), Self-correct, Confidence Check, Judge, and more.
- **Multiple Model Backends**: Ollama (local), Gemini API, and ScaleDown token compression wrapper.
- **Environment-Based Configuration**: Single .env file controls all model assignments and settings.
- **Singular Answer Prompting**: Optimized for concise, one-word/phrase responses to reduce fluff.
- **Binary Answer Checking**: Direct comparison against ground truth without judge calls.
- **Early Exit Policies**: Oracle or LLM-based gates for efficiency.
- **Debugging**: Verbose debug output with human-readable stage names and model information.
- **Real Dataset Support**: SimpleQA and TruthfulQA benchmark datasets included.

## Getting Started

### Requirements

- Python 3.8+
- `uv` or `pip`
- **Optional**: [ScaleDown](https://scaledown.xyz/) API key for token compression
- **Optional**: Google Gemini API key for Gemini models
- **Optional**: Local [Ollama](https://ollama.com/) installation for local models

Install dependencies:
```bash
# Using uv
uv pip install -r requirements.txt

# Or using pip
pip install requests python-dotenv
```

### Environment Variables

Create a `.env` file in the root directory with these variables:

#### Model Configuration
```env
# Model types for each role: ollama, gemini, scaledown
TARGET_MODEL_TYPE=ollama
HELPER_MODEL_TYPE=ollama  
JUDGE_MODEL_TYPE=ollama

# Ollama settings (if using ollama model type)
OLLAMA_MODEL=gemma3:27b
OLLAMA_ENDPOINT=http://localhost:11434/api/generate

# Gemini settings (if using gemini or scaledown model type)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash

# ScaleDown settings (if using scaledown model type)
SCALEDOWN_API_KEY=your_scaledown_api_key
TARGET_BASE_MODEL=gemini    # Base model for scaledown: gemini or ollama
HELPER_BASE_MODEL=gemini
JUDGE_BASE_MODEL=gemini
SD_RATE_TARGET=0.7          # Compression rates (0.0-1.0)
SD_RATE_HELPER=0.6
SD_RATE_JUDGE=0.0
```

#### Other Settings
```env
PIPE_DEBUG=1                              # Enable debug output
PIPE_DEBUG_MAXLEN=220                     # Max length for debug snippets
DATASET_PATH=data/simple_qa_test_set.csv  # Dataset file or "internal"
```

### Running

Ensure your `.env` file is correctly configured. The pipeline will display model configuration at startup:

```
=== Model Configuration ===
Target: ollama:gemma3:27b
Helper: scaledown(gemini:gemini-1.5-flash)
Judge: gemini:gemini-1.5-flash
```

Run the pipeline:
```bash
# Using uv
uv run main.py

# Or using standard python
python main.py
```

Sample output:
```
=== Dataset Question 1 :: 'Who received the IEEE Frank Rosenblatt Award in 2010?' - "Geoffrey Hinton" ===

[1] Q: Who received the IEEE Frank Rosenblatt Award in 2010?
  Final: Geoffrey Hinton
  Exit at: Completed All Stages
  Tokens: 42
  Correct: True
```

## Model Types

The pipeline supports three model types for each role (target/helper/judge):

- **`ollama`**: Direct connection to local Ollama API
- **`gemini`**: Direct connection to Google Gemini API
- **`scaledown`**: ScaleDown token compression → Base model (Gemini or Ollama)

### ScaleDown Integration

ScaleDown is used purely as a **token compression wrapper**:
- Compresses prompts before sending to base model
- Reduces token usage and API costs
- Supports wrapping any base model (Gemini or Ollama)
- Compression rates: 0.0 (no compression) to 1.0 (maximum compression)

### Pipeline Stages

Current default pipeline configuration:

```python
config = {
    "stages": [
        {"type":"baseline","id":"s0","model":"target"},           # Basic prompting
        {"type":"apo","id":"s1","helper":"helper","target":"target"}, # Prompt optimization
        {"type":"cove","id":"s2","model":"target"},               # Chain of verification
        {"type":"self_correct","id":"s3","model":"target"},       # Self-correction
        {"type":"confidence_check","id":"s4","model":"target"},   # Confidence assessment
    ],
    "gate": {"mode": "oracle"},  # Uses ground truth for early exit
    "token_diffs": True
}
```

### Stage Types Available

- **`baseline`**: Direct question → answer prompting
- **`apo`**: Automatic Prompt Optimization (helper rewrites prompt, target answers)
- **`cove`**: Chain-of-Verification (fact-checking previous answer)  
- **`self_correct`**: Self-correction of previous answer
- **`confidence_check`**: Tests model confidence when shown correct answer
- **`judge`**: External model judges answer correctness

## Key Changes in This Session

### 1. **Singular Answer Prompting**
- Updated all prompt templates to request concise, one-word/phrase responses
- Eliminates verbose LLM responses to focus on core answers

### 2. **Binary Answer Checking** 
- Direct string comparison against ground truth answers
- Removed dependency on LLM judge calls for basic correctness checking
- Uses oracle gate mode for early exit based on ground truth

### 3. **Clean Model Architecture**
- Removed `ScaledownDirectModel` - now ScaleDown is purely a compression wrapper  
- Three model types: `ollama` (local), `gemini` (API), `scaledown` (compression wrapper)
- Environment-based configuration for all model assignments

### 4. **Enhanced Debug Output**
- Shows model configuration at startup
- Human-readable stage technique names (e.g., "APO (Automatic Prompt Optimization)")
- Dataset questions display correct answers alongside questions
- Descriptive exit information instead of just stage IDs

### 5. **Confidence Check Stage**
- New stage that reveals correct answer and tests model honesty
- Forces choice between claiming confidence or admitting uncertainty
- Tracks whether models can assess their own confidence accurately

### 6. **Environment Variable Configuration**
```env
TARGET_MODEL_TYPE=ollama    # Target model for all answer stages
HELPER_MODEL_TYPE=gemini    # Helper model for APO prompt rewriting  
JUDGE_MODEL_TYPE=scaledown  # Judge model for evaluation stages
```

## File Overview

- `main.py` - Entry point with environment-based model configuration
- `pipeline.py` - Pipeline runner with updated debug output and templates
- `stages.py` - Stage implementations including new confidence_check stage
- `models.py` - Clean model adapters (Ollama, Gemini, ScaleDown wrapper)
- `dataset.py` - Dataset loading with SimpleQA and TruthfulQA support  
- `utils.py` - Utilities and data structures
- `data/` - Benchmark datasets (SimpleQA, TruthfulQA)

## Extending

- Add new stages by implementing the `Stage` protocol and registering via `@register_stage`
- Add new models by implementing the `Model` protocol in `models.py`  
- Modify prompt templates in the `DEFAULT_TEMPLATES` dictionary
- Create custom model configurations using the environment variable system

## License

MIT License

---

**Contact:** For ScaleDown access or model endpoints, visit [scaledown.xyz](https://scaledown.xyz/).
