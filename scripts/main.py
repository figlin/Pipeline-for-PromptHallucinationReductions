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
    SCALEDOWN_CHAT_URL = os.getenv("SCALEDOWN_CHAT_URL")  # your ScaleDown generation endpoint

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
                model=os.getenv("SD_MODEL_TARGET", "gemini/gemini-2.0-flash"),
                rate=float(os.getenv("SD_RATE_TARGET", "0.7")),
            ),
            "helper": ScaledownDirectModel(
                api_key=SCALEDOWN_API_KEY,
                endpoint=SCALEDOWN_CHAT_URL,
                model=os.getenv("SD_MODEL_HELPER", "gemini/gemini-2.0-flash"),
                rate=float(os.getenv("SD_RATE_HELPER", "0.6")),
            ),
            "judge": ScaledownDirectModel(
                api_key=SCALEDOWN_API_KEY,
                endpoint=SCALEDOWN_CHAT_URL,
                model=os.getenv("SD_MODEL_JUDGE", "gemini/gemini-2.0-flash"),
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

    gate_cfg = {"mode": "oracle"}
    if enable_judge_gate and not use_echo:
        gate_cfg = {"mode": "judge", "judge": "judge", "threshold": 0.8, "template": DEFAULT_TEMPLATES["gate_judge"]}

    # Budget and tracking configuration
    max_budget_usd = float(os.getenv("MAX_BUDGET_USD", "10.0"))
    enable_tracking = os.getenv("ENABLE_TRACKING", "1") not in ("0", "", "false", "False", "FALSE")
    db_path = os.getenv("TRACKING_DB_PATH", "usage_tracking.db")
    
    # Confidence thresholds for early exit
    confidence_thresholds = {
        "s0": float(os.getenv("BASELINE_EXIT_CONFIDENCE", "0.0")),  # Disabled by default
        "s1": float(os.getenv("APO_EXIT_CONFIDENCE", "0.0")),       # Disabled by default
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

    DEBUG = os.getenv("PIPE_DEBUG", "0") not in ("0", "", "false", "False", "FALSE")
    
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

    def print_pretty_trace(t):
        print("")
        print("=" * 80)
        print(f"Q[{t.qid}] {t.question}")
        print("-" * 80)
        for r in t.stage_results:
            print(f"[{r.stage_id}] confidence={r.confidence:.3f}")
            ev = r.evidence or {}
            for key in ("prompt", "helper_in", "optimized", "target_in", "verdict", "error"):
                if key in ev and ev[key]:
                    print(f"  {key}: {_snip(ev[key], pipeline.debug_maxlen)}")
            if r.answer is not None:
                print(f"  answer: {_snip(r.answer, pipeline.debug_maxlen)}")
            mu = r.model_usage or {}
            if isinstance(mu, dict) and ("prompt_tokens" in mu or "completion_tokens" in mu):
                pt = mu.get("prompt_tokens"); ct = mu.get("completion_tokens")
                model_name = mu.get("model")
                print(f"  usage: model={model_name} ptok={pt} ctok={ct}")
            elif isinstance(mu, dict):
                for sub, v in mu.items():
                    if isinstance(v, dict):
                        pt = v.get("prompt_tokens"); ct = v.get("completion_tokens")
                        model_name = v.get("model")
                        if pt or ct or model_name:
                            print(f"  usage.{sub}: model={model_name} ptok={pt} ctok={ct}")
        print("-" * 80)
        print(f"final: {_snip(t.final_answer, pipeline.debug_maxlen)}")
        print(f"exit_at: {t.early_exit_at}  tokens: {t.total_tokens}  cost: {t.total_cost:.6f}  time: {t.timing_sec:.3f}s")
        print("=" * 80)

    # Run pipeline
    logger.info("Starting pipeline run...")
    traces = []
    for ex in dataset:
        trace = pipeline.run_one(ex)
        traces.append(trace)
        
        if DEBUG:
            print_pretty_trace(trace)
        
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


