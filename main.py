from __future__ import annotations
from typing import Any, Dict
import os
import sys

from dotenv import load_dotenv
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import json

from dataset import EvalResult, dataset, load_from_csv
from evaluate import Evaluator
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
            sd_model = os.getenv(f"SD_LLM_MODEL_{role.upper()}", "gpt-4o") 
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

    DEBUG_CONTEXT = os.getenv("PIPE_DEBUG_CONTEXT", "0") not in ("0", "", "false", "False", "FALSE")
    
    config = {
        "stages": [
            {"type":"baseline","id":"s0","model":"target"}, # Target model baseline
            {"type":"apo","id":"s1","helper":"helper","target":"target"}, # Helper rewrites, target answers
            {"type":"cove","id":"s2","model":"target"}, # Target model verification
            {"type":"self_correct","id":"s3","model":"target"}, # Target model self-correction
            {"type":"judge","id":"s5","judge":"judge","exit_on_pass": True, "threshold": 0.8}
        ],
        "gate": {"mode": "oracle"},
        "token_diffs": True,
        "debug_context": DEBUG_CONTEXT
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

    # map stage id -> type for readable output
    stage_id_to_type = {s.get("id"): s.get("type") for s in config["stages"]}

    # ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # create a single log file per run (JSON Lines)
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_path = logs_dir / f"run-{run_ts}.jsonl"
    print(f"Logging to: {run_log_path}")

    # console verbosity for per-stage output: none | summary | full
    STAGE_CONSOLE = os.getenv("PIPE_STAGE_CONSOLE", "summary").lower()

    
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

    evaluator = Evaluator(dataset=dataset_to_run)
    total = EvalResult()
    for ex in dataset_to_run:
        trace = pipe.run_one(ex)
        # Use correct_answers for all datasets - SimpleQA populates this correctly
        golds = ex.correct_answers if ex.correct_answers else [ex.y_true]
        
        # Debug output
        if DEBUG:
            print(f"[DEBUG] Evaluating: pred='{trace.final_answer}' vs golds={golds}")
            print(f"[DEBUG] Schema: {evaluator.dataset.schema}, ex.y_true='{ex.y_true}', ex.correct_answers={ex.correct_answers}")
        
        result=evaluator._evaluate_example(
            pred=trace.final_answer,
            golds=golds,
        )
        print(f"[{ex.qid}] Q: {ex.question}")
        print("  Final:", trace.final_answer)
        print(" Result", result.model_dump())
        total.n += result.n
        total.em += result.em
        total.f1 += result.f1
        total.rouge1 += result.rouge1
        total.bleurt += result.bleurt
        total.llm_judge += result.llm_judge

        exit_technique = "Completed All Stages"
        if trace.early_exit_at:
            if trace.early_exit_at.startswith("gate_after:"):
                stage_id = trace.early_exit_at.split(":", 1)[1]
                exit_technique = f"Gate Exit after {get_stage_technique(stage_id, config['stages'])}"
            else:
                exit_technique = f"Stage Exit at {get_stage_technique(trace.early_exit_at, config['stages'])}"
        print("  Exit at:", exit_technique)
        print("  Tokens:", trace.total_tokens)

        # Per-stage console output (based on verbosity)
        if STAGE_CONSOLE in ("summary", "full"):
            for res in trace.stage_results:
                stype = stage_id_to_type.get(res.stage_id, "?")
                if STAGE_CONSOLE == "summary":
                    # concise one-liner per stage
                    mu = res.model_usage or {}
                    if "prompt_tokens" in mu or "completion_tokens" in mu:
                        pt, ct = mu.get("prompt_tokens"), mu.get("completion_tokens")
                        model = mu.get("model")
                    else:
                        # nested usage
                        sub_any = next(iter(mu.values()), {}) if isinstance(mu, dict) and mu else {}
                        pt, ct = (sub_any or {}).get("prompt_tokens"), (sub_any or {}).get("completion_tokens")
                        model = (sub_any or {}).get("model")
                    ans_snip = (res.answer or "").strip()
                    if len(ans_snip) > 120:
                        ans_snip = ans_snip[:117] + "..."
                    print(f"    - {res.stage_id} [{stype}] model={model} pt={pt} ct={ct} answer={ans_snip}")
                else:
                    # full details
                    print(f"    Stage {res.stage_id} ({stype})")
                    ev = res.evidence or {}
                    prompt_snip = ev.get("prompt") or ev.get("helper_in") or ev.get("target_in") or ""
                    if prompt_snip:
                        print("      prompt:", prompt_snip)
                    if "optimized" in ev:
                        print("      optimized:", ev.get("optimized"))
                    if res.answer is not None:
                        print("      answer:", res.answer)
                    mu = res.model_usage or {}
                    if mu:
                        if "prompt_tokens" in mu or "completion_tokens" in mu:
                            print("      usage:", {k: mu.get(k) for k in ("model","prompt_tokens","completion_tokens")})
                        else:
                            for label, sub in mu.items():
                                if isinstance(sub, dict):
                                    print(f"      usage.{label}:", {k: sub.get(k) for k in ("model","prompt_tokens","completion_tokens")})

        # Write this example to the single run log file (JSONL)
        try:
            with open(run_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[warn] failed to append log for {ex.qid}: {e}")

        # if ex.y_true:
        #     is_correct = trace.final_answer and trace.final_answer.strip().lower() == ex.y_true.lower()
        #     print("  Correct:", is_correct)

    total.em /= total.n
    total.f1 /= total.n
    total.rouge1 /= total.n
    total.bleu1 /= total.n
    total.llm_judge /= total.n
    total.mc_accuracy /= total.n
    print("Final Report: ", total.model_dump())
    print("-" * 50)
    
    if 'trace' in locals() and trace:
        print("Final:", trace.final_answer, "Exit at:", trace.early_exit_at, "Tokens:", trace.total_tokens)