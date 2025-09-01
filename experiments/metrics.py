from typing import Callable, List
import statistics as stats

def naive_binaryizer(metrics, threshold: float):
    print(metrics)
    f1 = metrics.get("f1") or 0.0
    print(f1)
    print("++++++")
    judge = metrics.get("llm_judge") or 0.0
    print(judge)
    print("============")
    print(threshold)
    return (f1 >= threshold) or (judge >= threshold)

def compute_binary_accuracy(traces: List, to_binary_fn: Callable, threshold: float):
    n_eval, n_ok = 0, 0
    for t in traces:
        y = to_binary_fn(metrics=t.final_metrics, threshold=threshold)
        n_eval += 1
        n_ok += int(y)
    return {"binary_accuracy": (n_ok / n_eval) if n_eval else 0.0, "n_evaluated": n_eval}

def summarize_budget(traces: List):
    toks = [t.total_tokens for t in traces]
    if not toks:
        return {"avg_tokens": 0.0, "p50_tokens": 0.0, "p90_tokens": 0.0}
    return {
        "avg_tokens": float(stats.mean(toks)),
        "p50_tokens": float(stats.median(toks)),
        "p90_tokens": float(stats.quantiles(toks, n=10)[8])  # ~p90
    }

def early_exit_hist(traces: List):
    from collections import Counter
    c = Counter(t.early_exit_at or "no_exit" for t in traces)
    return dict(c)
