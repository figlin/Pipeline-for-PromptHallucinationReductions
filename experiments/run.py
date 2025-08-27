import os, argparse, time
from typing import Dict, List

from data.loaders import load_simpleqa, load_truthfulqa
from experiments.logging import CsvLogger, write_summary, CsvStageLogger
from experiments.metrics import compute_binary_accuracy, summarize_budget, early_exit_hist, naive_binaryizer
from pipeline import build_pipeline

# Try ScaleDown direct model; fall back to echo so it runs without keys.
try:
    from models import ScaledownDirectModel
except Exception:
    ScaledownDirectModel = None

class EchoModel:
    name = "echo"
    def generate(self, prompt: str, temperature: float = 0.0, **kwargs):
        from utils import ModelResponse
        last = (prompt or "").splitlines()[-1]
        text = last if last else "echo"
        ptok = max(1, len(prompt)//4); ctok = max(1, len(text)//4)
        return ModelResponse(text=text, prompt_tokens=ptok, completion_tokens=ctok, cost=(ptok+ctok)*1e-6)

def build_models() -> Dict[str, object]:
    if ScaledownDirectModel and os.getenv("SCALEDOWN_API_KEY") and os.getenv("SCALEDOWN_CHAT_URL"):
        api = os.environ["SCALEDOWN_API_KEY"]; url = os.environ["SCALEDOWN_CHAT_URL"]
        return {
            "target": ScaledownDirectModel(api_key=api, endpoint=url,
                                           model=os.getenv("SD_MODEL_TARGET", "gemini/gemini-pro"),
                                           rate=float(os.getenv("SD_RATE_TARGET", "0.7"))),
            "helper": ScaledownDirectModel(api_key=api, endpoint=url,
                                           model=os.getenv("SD_MODEL_HELPER", "gemini/gemini-pro"),
                                           rate=float(os.getenv("SD_RATE_HELPER", "0.6"))),
            "judge": ScaledownDirectModel(api_key=api, endpoint=url,
                                          model=os.getenv("SD_MODEL_JUDGE", "gemini/gemini-pro"),
                                          rate=float(os.getenv("SD_RATE_JUDGE", "0.0")),
                                          default_params={"temperature": 0.0}),
        }
    e = EchoModel()
    return {"target": e, "helper": e, "judge": e}

def get_config(name: str):
    base = {"token_diffs": True, "gate": {"mode":"judge","judge":"judge","threshold":0.8}}
    if name == "baseline":
        stages=[{"type":"baseline","id":"s0","model":"target"}]
    elif name == "apo":
        stages=[{"type":"baseline","id":"s0","model":"target"},
                {"type":"apo","id":"s1","helper":"helper","target":"target"}]
    elif name == "apo_cove":
        stages=[{"type":"baseline","id":"s0","model":"target"},
                {"type":"apo","id":"s1","helper":"helper","target":"target"},
                {"type":"cove","id":"s2","model":"target"}]
    elif name == "apo_cove_sc":
        stages=[{"type":"baseline","id":"s0","model":"target"},
                {"type":"apo","id":"s1","helper":"helper","target":"target"},
                {"type":"cove","id":"s2","model":"target"},
                {"type":"self_correct","id":"s3","model":"target"}]
    else:  # full
        stages=[{"type":"baseline","id":"s0","model":"target"},
                {"type":"apo","id":"s1","helper":"helper","target":"target"},
                {"type":"cove","id":"s2","model":"target"},
                {"type":"self_correct","id":"s3","model":"target"},
                {"type":"judge","id":"s4","judge":"judge","exit_on_pass": True, "threshold":0.8}]
    return {**base, "stages": stages}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["simpleqa","truthfulqa"], required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--config", choices=["baseline","apo","apo_cove","apo_cove_sc","full"], default="full")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    data = load_simpleqa() if args.dataset=="simpleqa" else load_truthfulqa()
    if args.limit:
        data = data[:args.limit]

    models = build_models()
    pipe = build_pipeline(get_config(args.config), models)

    # enable console debug from env
    pipe.debug = os.getenv("PIPE_DEBUG", "0").lower() not in ("0", "false", "")
    pipe.debug_maxlen = int(os.getenv("PIPE_DEBUG_MAXLEN", "220"))

    traces = []
    t0 = time.time()
    for ex in data:
        tr = pipe.run_one(ex)
        tr.y_true = ex.y_true
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

    write_summary(out_dir, {
        "dataset": args.dataset,
        "config": args.config,
        "n": len(traces),
        **acc, **bud,
        "early_exit_histogram": hist,
        "runtime_sec": round(time.time() - t0, 2)
    })

    print(f"✅ Done: {out_dir}")
    print(f"  Accuracy: {acc['binary_accuracy']:.3f} on {acc['n_evaluated']} eval’d")
    print(f"  Avg tokens: {bud['avg_tokens']:.1f} | Early-exit: {hist}")

if __name__ == "__main__":
    main()
