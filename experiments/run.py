import os
import argparse
import time
import inspect
from typing import Dict, List, Optional

# ---- Use upstream dataset.py ------------------------------------------------

from dataset import Dataset


# ---- Dataset wrappers (return List[Example]) --------------------------------

def _load_ds(csv_path: str, limit: Optional[int], range_filter: Optional[str]):
    ds = Dataset(csv_path, range_filter=range_filter)
    exs = ds.examples
    if limit:
        exs = exs[: int(limit)]
    return exs

def load_simpleqa(limit: Optional[int] = None, range_filter: Optional[str] = None, **_) -> List:
    return _load_ds("data/simple_qa_test_set.csv", limit, range_filter)

def load_truthfulqa(limit: Optional[int] = None, range_filter: Optional[str] = None, **_) -> List:
    return _load_ds("data/TruthfulQA.csv", limit, range_filter)


# ---- Evaluator construction (robust to API changes) -------------------------

class DummyEvaluator:
    """Safe fallback so pipeline doesn't crash if evaluate.py API changes."""
    dataset = None  # will be set later

    def _evaluate_example(self, *args, **kwargs):
        # Minimal metrics dict; pipeline can still aggregate/print.
        return {
            "em": 0.0,
            "f1": 0.0,
            "rouge1": 0.0,
            "bleurt": 0.0,
            "llm_judge": 0.0,
            "mc_accuracy": 0.0,
        }

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

def _build_eval(models: Dict[str, object]):
    try:
        import evaluate as EV  # provided by the repo
    except Exception:
        return DummyEvaluator()

    # Candidates: function builders or class constructors
    candidates = []
    for name in ("build_evaluator", "get_evaluator", "make_evaluator"):
        if hasattr(EV, name) and callable(getattr(EV, name)):
            candidates.append(getattr(EV, name))
    if hasattr(EV, "Evaluator") and inspect.isclass(getattr(EV, "Evaluator")):
        candidates.append(getattr(EV, "Evaluator"))

    # Try each with multiple signatures
    for ctor in candidates:
        try:
            ev = _try_call(ctor, models)
            if ev is not None and hasattr(ev, "_evaluate_example"):
                return ev
        except Exception:
            continue

    # Final fallback
    return DummyEvaluator()


# ---- Analytics: logging & metrics ------------------------------------------

from experiments.logging import CsvLogger, write_summary, CsvStageLogger
from experiments.metrics import (
    compute_binary_accuracy,
    summarize_budget,
    early_exit_hist,
    naive_binaryizer,
)

from pipeline import build_pipeline

# Try ScaleDown direct model; fall back to echo so it runs without keys.
try:
    from models import ScaledownDirectModel  # type: ignore
except Exception:
    ScaledownDirectModel = None  # type: ignore


class EchoModel:
    name = "echo"

    def generate(self, prompt: str, temperature: float = 0.0, **kwargs):
        from utils import ModelResponse  # import lazily to avoid hard dep at import-time

        last = (prompt or "").splitlines()[-1]
        text = last if last else "echo"
        ptok = max(1, len(prompt) // 4)
        ctok = max(1, len(text) // 4)
        return ModelResponse(
            text=text,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            cost=(ptok + ctok) * 1e-6,
        )


def build_models() -> Dict[str, object]:
    if ScaledownDirectModel and os.getenv("SCALEDOWN_API_KEY") and os.getenv("SCALEDOWN_CHAT_URL"):
        api = os.environ["SCALEDOWN_API_KEY"]
        url = os.environ["SCALEDOWN_CHAT_URL"]
        return {
            "target": ScaledownDirectModel(
                api_key=api,
                endpoint=url,
                model=os.getenv("SD_MODEL_TARGET", "gemini/gemini-pro"),
                rate=float(os.getenv("SD_RATE_TARGET", "0.7")),
            ),
            "helper": ScaledownDirectModel(
                api_key=api,
                endpoint=url,
                model=os.getenv("SD_MODEL_HELPER", "gemini/gemini-pro"),
                rate=float(os.getenv("SD_RATE_HELPER", "0.6")),
            ),
            "judge": ScaledownDirectModel(
                api_key=api,
                endpoint=url,
                model=os.getenv("SD_MODEL_JUDGE", "gemini/gemini-pro"),
                rate=float(os.getenv("SD_RATE_JUDGE", "0.0")),
                default_params={"temperature": 0.0},
            ),
        }
    e = EchoModel()
    return {"target": e, "helper": e, "judge": e}


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
    ap.add_argument("--dataset", choices=["simpleqa", "truthfulqa"], required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--config", choices=["baseline", "apo", "apo_cove", "apo_cove_sc", "full"], default="full")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--range", default=None, help="Optional dataset range like '1-50' (overrides DATASET_RANGE)")
    args = ap.parse_args()

    range_filter = args.range or os.getenv("DATASET_RANGE")

    if args.dataset == "simpleqa":
        data = load_simpleqa(limit=args.limit, range_filter=range_filter)
    else:
        data = load_truthfulqa(limit=args.limit, range_filter=range_filter)

    models = build_models()
    cfg = get_config(args.config)

    evaluator = _build_eval(models)
    # Ensure evaluator.dataset.schema exists (pipeline expects it)
    class _SchemaBox:
        def __init__(self, schema: str):
            self.schema = schema

    need_set = (
        not hasattr(evaluator, "dataset")
        or evaluator.dataset is None
        or isinstance(evaluator.dataset, dict)
        or not hasattr(evaluator.dataset, "schema")
        or evaluator.dataset.schema is None  # type: ignore[attr-defined]
    )
    if need_set:
        try:
            evaluator.dataset = _SchemaBox(args.dataset)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Prefer new API (cfg, models, evaluator); fallback to old API (cfg, models)
    try:
        pipe = build_pipeline(cfg, models, evaluator)
    except TypeError:
        pipe = build_pipeline(cfg, models)

    # Enable console debug from env (best-effort)
    try:
        setattr(pipe, "debug", os.getenv("PIPE_DEBUG", "0").lower() not in ("0", "false", ""))
        setattr(pipe, "debug_maxlen", int(os.getenv("PIPE_DEBUG_MAXLEN", "220")))
    except Exception:
        pass

    traces = []
    t0 = time.time()
    for ex in data:
        tr = pipe.run_one(ex)
        tr.y_true = ex.y_true  # for downstream metrics
        traces.append(tr)

    out_dir = f"{args.out}/{args.dataset}/{args.config}"
    sample_logger = CsvLogger(out_dir)
    stage_logger = CsvStageLogger(out_dir)
    for tr in traces:
        sample_logger.log_trace(tr)
        stage_logger.log_trace(tr)
    sample_logger.flush()
    stage_logger.flush()

    acc = compute_binary_accuracy(traces, naive_binaryizer)
    bud = summarize_budget(traces)
    hist = early_exit_hist(traces)

    write_summary(
        out_dir,
        {
            "dataset": args.dataset,
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
