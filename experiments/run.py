import os
import argparse
import time
from typing import Dict

from dotenv import load_dotenv

from dataset import load_from_csv
from experiments.logging import CsvLogger, write_summary, CsvStageLogger
from experiments.metrics import (
    compute_binary_accuracy,
    summarize_budget,
    early_exit_hist,
    naive_binaryizer,
)

from evaluate import Evaluator
from main import create_model
from models import Model
from pipeline import build_pipeline

load_dotenv()
    
# Model type configuration from env
TARGET_MODEL_TYPE = os.getenv("TARGET_MODEL_TYPE", "ollama")  # ollama, gemini, scaledown, scaledown-llm
HELPER_MODEL_TYPE = os.getenv("HELPER_MODEL_TYPE", "ollama")
JUDGE_MODEL_TYPE = os.getenv("JUDGE_MODEL_TYPE", "ollama")

def _try_call(ctor, models):
    """Try a few common signatures for functions/classes."""
    try:
        return ctor(models=models)
    except TypeError:
        pass
    try:
        return ctor(models)
    except TypeError:
        pass
    try:
        return ctor()
    except TypeError:
        pass
    return None


def get_config(name: str):
    base = {"token_diffs": True, "gate": {"mode": "judge", "judge": "judge", "threshold": 0.8}}
    if name == "baseline":
        stages = [{"type": "baseline", "id": "s0", "model": "target"}]
    elif name == "apo":
        stages = [
            {"type": "baseline", "id": "s0", "model": "target"},
            {"type": "apo", "id": "s1", "helper": "helper", "target": "target"},
        ]
    elif name == "apo_cove":
        stages = [
            {"type": "baseline", "id": "s0", "model": "target"},
            {"type": "apo", "id": "s1", "helper": "helper", "target": "target"},
            {"type": "cove", "id": "s2", "model": "target"},
        ]
    elif name == "apo_cove_sc":
        stages = [
            {"type": "baseline", "id": "s0", "model": "target"},
            {"type": "apo", "id": "s1", "helper": "helper", "target": "target"},
            {"type": "cove", "id": "s2", "model": "target"},
            {"type": "self_correct", "id": "s3", "model": "target"},
        ]
    else:  # full
        stages = [
            {"type": "baseline", "id": "s0", "model": "target"},
            {"type": "apo", "id": "s1", "helper": "helper", "target": "target"},
            {"type": "cove", "id": "s2", "model": "target"},
            {"type": "self_correct", "id": "s3", "model": "target"},
            {"type": "judge", "id": "s4", "judge": "judge", "exit_on_pass": True, "threshold": 0.8},
        ]
    return {**base, "stages": stages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True, help="Path to dataset CSV file or HF name with base repo such as base_repo/dataset_name")
    ap.add_argument("--limit", type=int, help="Limit number of examples to run. if not passed we evaluate the whole dataset. Minimum limit is 2.")
    ap.add_argument("--config", choices=["baseline", "apo", "apo_cove", "apo_cove_sc", "full"], default="full")
    ap.add_argument("--threshold", type=float, default=0.8, help="Threshold for binarizer")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--range", default=None, help="Optional dataset range like '1-50' (overrides DATASET_RANGE)")
    args = ap.parse_args()
    
    dataset_to_run = load_from_csv(csv_path=args.dataset_path, range_filter=args.limit)

    print(len(dataset_to_run))

    models: Dict[str, Model] = {
        "target": create_model(TARGET_MODEL_TYPE, "target"),
        "helper": create_model(HELPER_MODEL_TYPE, "helper"), 
        "judge": create_model(JUDGE_MODEL_TYPE, "judge"),
    }

    evaluator = Evaluator(dataset=dataset_to_run, llm_judge_model=models.get("judge"))

    cfg = get_config(args.config)

    # Prefer new API (cfg, models, evaluator); fallback to old API (cfg, models)
    pipe = build_pipeline(cfg, models, evaluator)

    # Enable console debug from env (best-effort)
    try:
        setattr(pipe, "debug", os.getenv("PIPE_DEBUG", "0").lower() not in ("0", "false", ""))
        setattr(pipe, "debug_maxlen", int(os.getenv("PIPE_DEBUG_MAXLEN", "220")))
    except Exception:
        pass

    traces = []
    t0 = time.time()
    for ex in dataset_to_run:
        tr = pipe.run_one(ex)
        traces.append(tr)

    out_dir = f"{args.out}/{os.path.basename(args.dataset_path).split('.')[0]}/{args.config}"
    sample_logger = CsvLogger(out_dir)
    stage_logger = CsvStageLogger(out_dir)
    for tr in traces:
        sample_logger.log_trace(tr)
        stage_logger.log_trace(tr)
    sample_logger.flush()
    stage_logger.flush()

    acc = compute_binary_accuracy(
        traces=traces,
        to_binary_fn=naive_binaryizer,
        threshold=args.threshold)
    bud = summarize_budget(traces)
    hist = early_exit_hist(traces)

    write_summary(
        out_dir,
        {
            "dataset": args.dataset_path,
            "config": args.config,
            "n": len(traces),
            **acc,
            **bud,
            "early_exit_histogram": hist,
            "runtime_sec": round(time.time() - t0, 2),
        },
    )

    print(f"✅ Done: {out_dir}")
    print(f"  Accuracy: {acc['binary_accuracy']:.3f} on {acc['n_evaluated']} eval’d")
    print(f"  Avg tokens: {bud['avg_tokens']:.1f} | Early-exit: {hist}")


if __name__ == "__main__":
    main()
