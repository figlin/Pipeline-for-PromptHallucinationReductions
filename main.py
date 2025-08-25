from __future__ import annotations
from typing import Any, Dict
import os
import sys

from dotenv import load_dotenv

from dataset import Dataset, dataset, load_from_csv
from models import Model, ScaleDownCompressionWrapper, ScaleDownLLMWrapper, GeminiModel, OllamaModel
from pipeline import DEFAULT_TEMPLATES, build_pipeline

load_dotenv()

# -------------------------
# Minimal runnable demo
# -------------------------

if __name__ == "__main__":
    # Read env
    SCALEDOWN_API_KEY = os.environ.get("SCALEDOWN_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    
    # Model type configuration from env
    TARGET_MODEL_TYPE = os.getenv("TARGET_MODEL_TYPE", "ollama")  # ollama, gemini, scaledown, scaledown-llm
    HELPER_MODEL_TYPE = os.getenv("HELPER_MODEL_TYPE", "ollama")
    JUDGE_MODEL_TYPE = os.getenv("JUDGE_MODEL_TYPE", "ollama")
    
    # Ollama settings
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")
    OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
    
    # Gemini settings  
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    def get_stage_technique(stage_id: str, stages_config: list) -> str:
        """Get the optimization technique name for a stage"""
        stage_map = {
            "baseline": "Baseline (No Optimization)",
            "apo": "APO (Automatic Prompt Optimization)",
            "cove": "CoVe (Chain-of-Verification)",
            "self_correct": "Self-Correction",
            "confidence_check": "Confidence Check",
            "judge": "Judge Evaluation"
        }
        
        for stage in stages_config:
            if stage.get("id") == stage_id:
                return stage_map.get(stage["type"], f"Unknown ({stage['type']})")
        return "Unknown Stage"

    def create_model(model_type: str, role: str) -> Model:
        """Create a model based on type and role (target/helper/judge)"""
        if model_type == "ollama":
            return OllamaModel(
                model=OLLAMA_MODEL,
                endpoint=OLLAMA_ENDPOINT,
                timeout=240.0,
            )
        elif model_type == "gemini":
            temp = 0.2 if role == "helper" else 0.0
            return GeminiModel(
                api_key=GEMINI_API_KEY,
                model=GEMINI_MODEL,
                default_params={"temperature": temp}
            )
        elif model_type == "scaledown":
            # ScaleDown compression wrapper - can wrap any base model - defaults to Gemini
            base_model_type = os.getenv(f"{role.upper()}_BASE_MODEL", "gemini")  # gemini or ollama
            
            if base_model_type == "gemini":
                temp = 0.2 if role == "helper" else 0.0
                base_model = GeminiModel(
                    api_key=GEMINI_API_KEY,
                    model=GEMINI_MODEL,
                    default_params={"temperature": temp}
                )
            elif base_model_type == "ollama":
                base_model = OllamaModel(
                    model=OLLAMA_MODEL,
                    endpoint=OLLAMA_ENDPOINT,
                    timeout=240.0,
                )
            else:
                raise ValueError(f"Unknown base model type for ScaleDown: {base_model_type}")
                
            rate = float(os.getenv(f"SD_RATE_{role.upper()}", "0.7"))
            return ScaleDownCompressionWrapper(
                base=base_model,
                api_key=SCALEDOWN_API_KEY,
                rate=rate,
            )
        elif model_type == "scaledown-llm":
            # ScaleDown LLM wrapper - direct LLM prompting via ScaleDown API
            sd_model = os.getenv(f"SD_LLM_MODEL_{role.upper()}", "gpt-4o")  # Default to GPT-4o
            rate = float(os.getenv(f"SD_LLM_RATE_{role.upper()}", "0.7"))
            temp = 0.2 if role == "helper" else 0.0
            
            return ScaleDownLLMWrapper(
                api_key=SCALEDOWN_API_KEY,
                model=sd_model,
                rate=rate,
                default_params={"temperature": temp}
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    models: Dict[str, Model] = {
        "target": create_model(TARGET_MODEL_TYPE, "target"),
        "helper": create_model(HELPER_MODEL_TYPE, "helper"), 
        "judge": create_model(JUDGE_MODEL_TYPE, "judge"),
    }

    config = {
        "stages": [
            {"type":"baseline","id":"s0","model":"target"}, # Target model baseline
            {"type":"apo","id":"s1","helper":"helper","target":"target"}, # Helper rewrites, target answers
            {"type":"cove","id":"s2","model":"target"}, # Target model verification
            {"type":"self_correct","id":"s3","model":"target"}, # Target model self-correction
            {"type":"confidence_check","id":"s4","model":"target"}, # Target model confidence check
            # {"type":"judge","id":"s5","judge":"judge","exit_on_pass": True, "threshold": 0.8}
        ],
        "gate": {"mode": "oracle"},
        "token_diffs": True
    }

    DEBUG = os.getenv("PIPE_DEBUG", "0") not in ("0", "", "false", "False", "FALSE")
    # Print model configuration
    print("=== Model Configuration ===")
    print(f"Target: {models['target'].name}")
    print(f"Helper: {models['helper'].name}")
    print(f"Judge: {models['judge'].name}")
    print()

    pipe = build_pipeline(config, models)
    # re-wrap with debug (or pass debug into a factory if you prefer)
    pipe.debug = DEBUG
    pipe.debug_maxlen = int(os.getenv("PIPE_DEBUG_MAXLEN") or "220")

    
    # Load dataset based on environment variable
    dataset_path = os.getenv("DATASET_PATH", "internal")
    dataset_range = os.getenv("DATASET_RANGE", "")
    
    if dataset_path.lower() == "internal":
        dataset_to_run = dataset
        print("Running with internal dataset.")
    else:
        print(f"Running with dataset from: {dataset_path}")
        if dataset_range:
            print(f"Using dataset range: {dataset_range}")
        dataset_to_run = load_from_csv(dataset_path, dataset_range if dataset_range else None)

    if not dataset_to_run:
        print(f"Error: No data loaded from '{dataset_path}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    for ex in dataset_to_run:
        trace = pipe.run_one(ex)
        print(f"[{ex.qid}] Q: {ex.question}")
        print("  Final:", trace.final_answer)
        exit_technique = "Completed All Stages"
        if trace.early_exit_at:
            if trace.early_exit_at.startswith("gate_after:"):
                stage_id = trace.early_exit_at.split(":", 1)[1]
                exit_technique = f"Gate Exit after {get_stage_technique(stage_id, config['stages'])}"
            else:
                exit_technique = f"Stage Exit at {get_stage_technique(trace.early_exit_at, config['stages'])}"
        print("  Exit at:", exit_technique)
        print("  Tokens:", trace.total_tokens)
        if ex.y_true:
            is_correct = trace.final_answer and trace.final_answer.strip().lower() == ex.y_true.lower()
            print("  Correct:", is_correct)
        print("-" * 50)
    
    if 'trace' in locals() and trace:
        print("Final:", trace.final_answer, "Exit at:", trace.early_exit_at, "Tokens:", trace.total_tokens)

