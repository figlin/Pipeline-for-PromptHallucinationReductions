from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol, Callable
import difflib
import time
import uuid
import os
import requests
import json
from pathlib import Path
from datetime import datetime

from src.core.dataset_loader import Example, dataset
from src.core.models import Model, ScaledownDirectModel, EchoModel
from src.core.pipeline import build_pipeline
from src.core.templates import DEFAULT_TEMPLATES
from src.analytics.tracking import UsageTracker
from src.analytics.analysis import PipelineAnalyzer
from src.analytics.visualization import create_visualization_report
import logging
import sys
from dotenv import load_dotenv
load_dotenv()

# -------------------------
# Minimal runnable demo
# -------------------------

if __name__ == "__main__":
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    # Prepare runs directory and file logging
    runs_dir = Path(os.getenv("RUNS_DIR", "runs"))
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = runs_dir / f"pipeline_{run_id}.log"
    tracefile = runs_dir / f"trace_{run_id}.jsonl"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level, logging.INFO))
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    logger = logging.getLogger("main")
    use_echo = os.getenv("USE_ECHO", "0") not in ("0", "", "false", "False", "FALSE")
    SCALEDOWN_API_KEY = os.getenv("SCALEDOWN_API_KEY")
    SCALEDOWN_CHAT_URL = os.getenv("SCALEDOWN_CHAT_URL")  # your ScaleDown compression endpoint

    if use_echo:
        echo = EchoModel()
        models: Dict[str, Model] = {"target": echo, "helper": echo, "judge": echo}
    else:
        if not SCALEDOWN_API_KEY or not SCALEDOWN_CHAT_URL:
            raise RuntimeError("Missing SCALEDOWN_API_KEY or SCALEDOWN_CHAT_URL. Set USE_ECHO=1 for local runs.")
        models: Dict[str, Model] = {
            "target": ScaledownDirectModel(
                api_key=SCALEDOWN_API_KEY,
                endpoint=SCALEDOWN_CHAT_URL,
                model=os.getenv("SD_MODEL_TARGET", "gemini-2.5-flash"),
                rate=float(os.getenv("SD_RATE_TARGET", "0.7")),
            ),
            "helper": ScaledownDirectModel(
                api_key=SCALEDOWN_API_KEY,
                endpoint=SCALEDOWN_CHAT_URL,
                model=os.getenv("SD_MODEL_HELPER", "gemini-2.5-flash"),
                rate=float(os.getenv("SD_RATE_HELPER", "0.6")),
            ),
            "judge": ScaledownDirectModel(
                api_key=SCALEDOWN_API_KEY,
                endpoint=SCALEDOWN_CHAT_URL,
                model=os.getenv("SD_MODEL_JUDGE", "gemini-2.5-flash"),
                rate=float(os.getenv("SD_RATE_JUDGE", "0.0")),
                default_params={"temperature": 0.0},
            ),
        }

    enable_judge_stage = os.getenv("ENABLE_JUDGE_STAGE", "0") not in ("0", "", "false", "False", "FALSE")
    enable_judge_gate = os.getenv("ENABLE_JUDGE_GATE", "0") not in ("0", "", "false", "False", "FALSE")

    stages_cfg = [
        {"type":"baseline","id":"s0","model":"target"},
        {"type":"apo","id":"s1","helper":"helper","target":"target"},
        {"type":"cove","id":"s2","model":"target"},
        {"type":"self_correct","id":"s3","model":"target"},
    ]
    if enable_judge_stage and not use_echo:
        stages_cfg.append({"type":"judge","id":"s4","judge":"judge","exit_on_pass": True, "threshold": 0.8})

    gate_cfg = {"mode": "none"}  # Disable judge gate - only use confidence-based early exit

    # Budget and tracking configuration
    max_budget_usd = float(os.getenv("MAX_BUDGET_USD", "10.0"))
    enable_tracking = os.getenv("ENABLE_TRACKING", "1") not in ("0", "", "false", "False", "FALSE")
    db_path = os.getenv("TRACKING_DB_PATH", "usage_tracking.db")
    
    # Confidence thresholds for early exit
    confidence_thresholds = {
        "s0": float(os.getenv("BASELINE_EXIT_CONFIDENCE", "0.0")),  # Disabled - baseline should never exit early
        "s1": float(os.getenv("APO_EXIT_CONFIDENCE", "0.8")),       # Enable early exit after APO with high confidence
        "s2": float(os.getenv("COVE_EXIT_CONFIDENCE", "0.9")),      # High confidence for CoVe
        "s3": float(os.getenv("SELF_CORRECT_EXIT_CONFIDENCE", "0.0"))  # Disabled by default
    }
    
    # Update stage configs with confidence thresholds
    for stage in stages_cfg:
        stage_id = stage["id"]
        if stage_id in confidence_thresholds and confidence_thresholds[stage_id] > 0:
            stage["exit_confidence"] = confidence_thresholds[stage_id]
    
    config = {
        "stages": stages_cfg,
        "gate": gate_cfg,
        "token_diffs": True,
        "max_budget_usd": max_budget_usd
    }

    DEBUG = os.getenv("PIPE_DEBUG", "1") not in ("0", "", "false", "False", "FALSE")
    
    # Initialize usage tracker
    usage_tracker = None
    if enable_tracking:
        usage_tracker = UsageTracker(max_budget_usd=max_budget_usd, db_path=db_path)
        logger.info(f"Initialized usage tracker with budget: ${max_budget_usd:.2f}")
    
    # Initialize pipeline
    pipeline = build_pipeline(config, models, usage_tracker=usage_tracker)
    pipeline.debug = DEBUG
    pipeline.debug_maxlen = int(os.getenv("PIPE_DEBUG_MAXLEN", "220"))

    def _snip(s: Optional[str], n: int = 220) -> str:
        if not s:
            return ""
        s = str(s)
        return s if len(s) <= n else s[:n] + "..."

    def print_hallucination_reduction_pipeline(t):
        """Print a clear, readable view of the hallucination reduction pipeline"""
        print("\n" + "="*80)
        print(f"HALLUCINATION REDUCTION PIPELINE - Question {t.qid}")
        print("="*80)
        
        print(f"\nQUESTION: {t.question}")
        if hasattr(t, 'y_true') and t.y_true:
            print(f"EXPECTED ANSWER: {t.y_true}")
        
        print("\n" + "-"*80)
        print("PIPELINE STAGES & RESULTS:")
        print("-"*80)
        
        stage_names = {
            "s0": "BASELINE (Direct Answer)",
            "s1": "APO (Prompt Optimization)", 
            "s2": "CoVe (Chain-of-Verification)",
            "s3": "SELF-CORRECT (Error Fixing)",
            "s4": "JUDGE (Quality Assessment)"
        }
                
        for i, r in enumerate(t.stage_results):
            stage_name = stage_names.get(r.stage_id, f"Stage {r.stage_id}")
            confidence = r.confidence
            
            print(f"\n{stage_name}")
            print(f"   Confidence: {confidence:.3f}")
            
            ev = r.evidence or {}
            
            if "prompt" in ev:
                print(f"\n--- PROMPT ({stage_name}) ---")
                print(ev["prompt"])
                print("-" * 40)
            
            # Show answers / verdicts depending on stage
            if r.stage_id == "s0" and r.answer:
                print(f"   Answer: {r.answer}")
            elif r.stage_id == "s1":  # APO
                if "helper_in" in ev:
                    print("\n--- APO REWRITE PROMPT ---")
                    print(ev["helper_in"])
                    print("-" * 40)
                if "target_in" in ev:
                    print("\n--- APO TARGET PROMPT ---")
                    print(ev["target_in"])
                    print("-" * 40)
                if r.answer:
                    print(f"   Answer: {r.answer}")
                if "helper_error" in ev:
                    print(f"   Error: {ev['helper_error']}")
            elif r.stage_id == "s2" and r.answer:
                print(f"   Verified Answer: {r.answer}")
            elif r.stage_id == "s3" and r.answer:
                print(f"   Corrected Answer: {r.answer}")
            elif r.stage_id == "s4" and "verdict" in ev:
                print(f"   Verdict: {ev['verdict']}")
        
        print("\n" + "-"*80)
        print("FINAL RESULTS:")
        print("-"*80)
        
        # Check if answer is correct
        expected = getattr(t, 'y_true', None)
        final_answer = t.final_answer or "No answer generated"
        is_correct = expected and final_answer.strip().lower() == expected.lower()
        
        print(f"FINAL ANSWER: {final_answer}")
        if expected:
            print(f"CORRECTNESS: {'CORRECT' if is_correct else 'INCORRECT'} (Expected: {expected})")
        
        # Show pipeline efficiency
        if t.early_exit_at:
            print(f"EARLY EXIT: Pipeline stopped at stage {t.early_exit_at}")
            if t.early_exit_at.startswith("gate_after:"):
                print(f"   REASON: Gate triggered early exit after {t.early_exit_at.split(':')[1]}")
            else:
                print(f"   REASON: Stage {t.early_exit_at} requested early exit")
                # Show confidence threshold info
                if t.early_exit_at in ["s1", "s2", "s3"]:
                    stage_result = next((r for r in t.stage_results if r.stage_id == t.early_exit_at), None)
                    if stage_result:
                        print(f"   CONFIDENCE: {stage_result.confidence:.3f} (threshold met)")
        else:
            print(f"COMPLETION: All stages completed")
            
        print(f"COST: ${t.total_cost:.6f}")
        print(f"TIME: {t.timing_sec:.3f}s")
        print(f"TOKENS: {t.total_tokens}")
        
        # Show hallucination reduction effectiveness
        if len(t.stage_results) > 1:
            baseline_answer = t.stage_results[0].answer if t.stage_results[0].answer else "No baseline answer"
            final_answer = t.final_answer or "No final answer"
            
            print(f"\nHALLUCINATION REDUCTION ANALYSIS:")
            print(f"   Baseline: {baseline_answer}")
            print(f"   Final: {final_answer}")
            
            if baseline_answer != final_answer:
                print(f"   Pipeline modified the answer - potential hallucination reduction!")
            else:
                print(f"   Answer unchanged - baseline was already good")
        
        print("="*80 + "\n")

    # Run pipeline
    logger.info("Starting pipeline run...")
    traces = []
    for ex in dataset:
        trace = pipeline.run_one(ex)
        traces.append(trace)

        # -------------------------------
        # NEW: Immediate print of APO prompt
        # -------------------------------
        apo_stage = next((r for r in trace.stage_results if r.stage_id == "s1"), None)
        optimized_prompt = (apo_stage.evidence or {}).get("optimized") if apo_stage else None
        if optimized_prompt:
            print("\n" + "-"*80)
            print(f"QID {ex.qid} — Optimized Prompt (APO)")
            print("-"*80)
            print(optimized_prompt)
            print("-"*80 + "\n")
            logger.info("optimized_prompt qid=%s\n%s", ex.qid, optimized_prompt)

        if DEBUG:
            print_hallucination_reduction_pipeline(trace)
        
        corr = bool(trace.final_answer and trace.final_answer.strip().lower() == (ex.y_true or "").lower())
        logger.info("result qid=%s exit=%s tokens=%s correct=%s confidence=%.3f", 
                   ex.qid, trace.early_exit_at, trace.total_tokens, corr, 
                   max((r.confidence for r in trace.stage_results), default=0.0))
        
        # Check if budget exceeded
        if usage_tracker and not usage_tracker.check_budget():
            logger.warning("Budget exceeded, stopping pipeline")
            break
        
        try:
            with open(tracefile, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("failed.writing.trace jsonl")
    
    # Log pipeline run completion
    if usage_tracker:
        usage_tracker.log_pipeline_run(run_id, len(traces), config)
        usage_summary = usage_tracker.get_usage_summary()
        logger.info(f"Usage summary: {usage_summary['total_tokens']} tokens, ${usage_summary['total_cost_usd']:.6f} cost")
    
    logger.info(f"Pipeline run completed. Processed {len(traces)} examples.")
    
    # Print summary of hallucination reduction effectiveness
    if traces:
        print("\n" + "="*80)
        print("HALLUCINATION REDUCTION PIPELINE SUMMARY")
        print("="*80)
        
        total_questions = len(traces)
        correct_answers = sum(1 for t in traces if t.final_answer and t.y_true and 
                            t.final_answer.strip().lower() == t.y_true.lower())
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        print(f"\nACCURACY: {correct_answers}/{total_questions} ({accuracy:.1f}%)")
        
        # Show how many answers were modified by the pipeline
        modified_answers = 0
        for t in traces:
            if len(t.stage_results) > 1:
                baseline_answer = t.stage_results[0].answer if t.stage_results[0].answer else ""
                final_answer = t.final_answer or ""
                if baseline_answer != final_answer:
                    modified_answers += 1
        
        print(f"ANSWERS MODIFIED: {modified_answers}/{total_questions} ({modified_answers/total_questions*100:.1f}%)")
        
        # Show average confidence
        avg_confidence = sum(max((r.confidence for r in t.stage_results), default=0.0) 
                           for t in traces) / total_questions if total_questions > 0 else 0
        print(f"AVERAGE CONFIDENCE: {avg_confidence:.3f}")
        
        # Show cost and efficiency
        total_cost = sum(t.total_cost for t in traces)
        total_tokens = sum(t.total_tokens for t in traces)
        avg_time = sum(t.timing_sec for t in traces) / total_questions if total_questions > 0 else 0
        
        print(f"TOTAL COST: ${total_cost:.6f}")
        print(f"TOTAL TOKENS: {total_tokens}")
        print(f"AVERAGE TIME: {avg_time:.3f}s per question")
        
        print("\n" + "="*80)
    
    # Generate analysis and visualizations
    if enable_tracking and traces:
        logger.info("Generating analysis and visualizations...")
        
        # Run analysis
        analyzer = PipelineAnalyzer(db_path)
        analysis_results = analyzer.analyze_run_traces(traces)
        
        # Save analysis results
        analysis_file = runs_dir / f"analysis_{run_id}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Generate visualizations
        plots_dir = runs_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        saved_plots = create_visualization_report(str(tracefile), db_path, str(plots_dir))
        
        logger.info(f"Analysis saved to {analysis_file}")
        logger.info(f"Plots saved to {plots_dir}")
        
        # Print optimization recommendations
        recommendations = analysis_results.get("optimization_recommendations", [])
        if recommendations:
            logger.info("Optimization recommendations:")
            for rec in recommendations:
                logger.info(f"  - {rec.description}")
                logger.info(f"    Potential savings: {rec.potential_savings_tokens} tokens, ${rec.potential_savings_cost:.4f}")
        else:
            logger.info("No optimization recommendations at this time.")
