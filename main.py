from __future__ import annotations
from typing import Any, Dict
import os
import sys

from dotenv import load_dotenv

from dataset import AggregatedMetrics, dataset, load_from_csv, EvalResult
from evaluate import Evaluator
from models import Model, ScaleDownCompressionWrapper, ScaleDownLLMWrapper, GeminiModel, OllamaModel
from pipeline import DEFAULT_TEMPLATES, build_pipeline
from pipeline_config import get_config, list_configs
from utils import normalize_text

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

    DEBUG_CONTEXT = os.getenv("PIPE_DEBUG_CONTEXT", "0") not in ("0", "", "false", "False", "FALSE")
    
    # Load pipeline configuration
    config_name = os.getenv("PIPELINE_CONFIG", "default")
    
    # Special handling for listing configs
    if config_name == "list":
        list_configs()
        sys.exit(0)

    DEBUG = os.getenv("PIPE_DEBUG", "0") not in ("0", "", "false", "False", "FALSE")
    # Print model configuration
    print("=== Model Configuration ===")
    print(f"Target: {models['target'].name}")
    print(f"Helper: {models['helper'].name}")
    print(f"Judge: {models['judge'].name}")
    print()

    
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
    
    # Load pipeline configuration based on dataset or environment setting
    dataset_schema = getattr(dataset_to_run, 'schema', None) if dataset_to_run else None
    config = get_config(config_name, dataset_schema)
    
    # Override debug setting from environment
    config["debug_context"] = DEBUG_CONTEXT
    
    evaluator = Evaluator(dataset=dataset_to_run, llm_judge_model=models.get("judge"))
    
    print(f"Using pipeline configuration: '{config_name}' for dataset: {dataset_schema}")
    if config.get("gate", {}).get("mode") == "metric":
        print(f"Gate criteria: {config['gate']['criteria']}")
    print()
    
    pipe = build_pipeline(config, models, evaluator)
    # re-wrap with debug (or pass debug into a factory if you prefer)
    pipe.debug = DEBUG
    pipe.debug_maxlen = int(os.getenv("PIPE_DEBUG_MAXLEN") or "220")
    pipe.evaluate_all_stages = True  # Enable per-stage evaluation
    
    total = EvalResult()
    gate_exit_counts = {}  # Track where each prompt exited
    
    for ex in dataset_to_run:
        trace = pipe.run_one(ex)
        # Use correct_answers for all datasets - SimpleQA populates this correctly
        golds = ex.correct_answers if ex.correct_answers else [ex.y_true]
        
        # Debug output
        if DEBUG:
            print(f"[DEBUG] Evaluating: pred='{trace.final_answer}' vs golds={golds}")
            print(f"[DEBUG] Schema: {evaluator.dataset.schema}, ex.y_true='{ex.y_true}', ex.correct_answers={ex.correct_answers}")
            print(f"[DEBUG] Normalized pred: '{normalize_text(trace.final_answer)}' vs normalized golds: {[normalize_text(g) for g in golds]}")
        
        result=evaluator._evaluate_example(
            pred=trace.final_answer,
            golds=golds,
            bads=ex.incorrect_answers,
        )
        
        # Add result to total
        total.n += result.n
        total.em += result.em
        total.f1 += result.f1
        total.rouge1 += result.rouge1
        total.bleurt += result.bleurt
        total.llm_judge += result.llm_judge
        total.mc_accuracy += result.mc_accuracy
        
        expected = ex.y_true or (ex.correct_answers[0] if ex.correct_answers else "N/A")
        print(f"[{ex.qid}] Q: {ex.question}")
        print(f"  Expected: {expected}")
        print("  Final:", trace.final_answer)
        
        # Show stage-by-stage results for this question
        if hasattr(trace, 'stage_results') and trace.stage_results:
            print("  Stage Results:")
            for stage_result in trace.stage_results:
                stage_id = stage_result.stage_id
                answer = stage_result.answer or "<No answer>"
                # Calculate metrics for this single result by comparing with gold
                single_result = evaluator._evaluate_example(
                    pred=stage_result.answer,
                    golds=golds,
                    bads=ex.incorrect_answers,
                )
                print(f"    {stage_id}: {answer}")
                print(f"      EM={single_result.em:.3f}, F1={single_result.f1:.3f}, ROUGE1={single_result.rouge1:.3f}, BLEURT={single_result.bleurt:.3f}, LLM_Judge={single_result.llm_judge:.3f}, MC_Accuracy={single_result.mc_accuracy:.3f}")
        else:
            print("  (Stage-by-stage results not available)")
        
        exit_technique = "Completed All Stages"
        if trace.early_exit_at:
            if trace.early_exit_at.startswith("gate_after:"):
                stage_id = trace.early_exit_at.split(":", 1)[1]
                exit_technique = f"Gate Exit after {get_stage_technique(stage_id, config['stages'])}"
            else:
                exit_technique = f"Stage Exit at {get_stage_technique(trace.early_exit_at, config['stages'])}"
        
        # Track gate exit counts
        gate_exit_counts[exit_technique] = gate_exit_counts.get(exit_technique, 0) + 1
        
        print("  Exit at:", exit_technique)
        print("  Tokens:", trace.total_tokens)
        print()  # Add blank line for spacing
        # if ex.y_true:
        #     is_correct = trace.final_answer and trace.final_answer.strip().lower() == ex.y_true.lower()
        #     print("  Correct:", is_correct)

    if total.n > 0:
        total.em /= total.n
        total.f1 /= total.n
        total.rouge1 /= total.n
        total.bleurt /= total.n
        total.llm_judge /= total.n
        total.mc_accuracy /= total.n
    print("Final Report: ", total.model_dump())
    print("-" * 50)
    
    # Print per-stage metrics if available
    if hasattr(pipe, 'aggregated_metrics') and pipe.aggregated_metrics._results:
        print("\n=== Per-Stage Performance ===")
        for stage_name, stage_result in pipe.aggregated_metrics._results.items():
            if stage_result.n > 0:
                avg_result = EvalResult(
                    n=stage_result.n,
                    em=stage_result.em / stage_result.n,
                    f1=stage_result.f1 / stage_result.n,
                    rouge1=stage_result.rouge1 / stage_result.n,
                    bleurt=stage_result.bleurt / stage_result.n,
                    llm_judge=stage_result.llm_judge / stage_result.n,
                    mc_accuracy=stage_result.mc_accuracy / stage_result.n
                )
                print(f"{stage_name}: {avg_result.model_dump()}")
    
    # Print gate exit summary table
    print("\n=== Gate Exit Summary ===")
    if gate_exit_counts:
        # Calculate percentages and format table
        total_questions = sum(gate_exit_counts.values())
        
        # Sort by count (descending) for better readability
        sorted_exits = sorted(gate_exit_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Exit Point':<35} {'Count':<8} {'Percentage':<10}")
        print("-" * 55)
        
        for exit_point, count in sorted_exits:
            percentage = (count / total_questions) * 100
            print(f"{exit_point:<35} {count:<8} {percentage:>6.1f}%")
        
        print("-" * 55)
        print(f"{'Total Questions':<35} {total_questions:<8} {'100.0%':<10}")
        
        # Additional insights
        early_exits = total_questions - gate_exit_counts.get("Completed All Stages", 0)
        if early_exits > 0:
            early_exit_rate = (early_exits / total_questions) * 100
            print(f"\nEarly Exit Rate: {early_exit_rate:.1f}% ({early_exits}/{total_questions} questions)")
            
            # Calculate token savings (rough estimate)
            avg_tokens_early = sum(trace.total_tokens for trace in locals().get('traces', []) if trace.early_exit_at) / max(early_exits, 1) if early_exits > 0 else 0
            if 'trace' in locals() and trace:
                # Use the last trace as an estimate for full pipeline tokens
                estimated_full_tokens = trace.total_tokens if not trace.early_exit_at else trace.total_tokens * 1.5
                if avg_tokens_early > 0:
                    estimated_savings = (estimated_full_tokens - avg_tokens_early) / estimated_full_tokens * 100
                    print(f"Estimated Token Savings: ~{estimated_savings:.1f}% per early exit")
        else:
            print(f"\nEarly Exit Rate: 0.0% (No early exits - all questions completed all stages)")
    else:
        print("No gate exit data collected.")
    
    print()
    