# experiments/logging.py
import csv, json
from pathlib import Path
from typing import List, Dict, Any

class CsvLogger:
    """Per-sample CSV + run summary (unchanged behavior)."""
    def __init__(self, out_dir: str):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows: List[dict] = []

    def log_trace(self, t) -> None:
        self.rows.append({
            "qid": t.qid,
            "question": t.question,
            "final_answer": t.final_answer or "",
            "early_exit_at": t.early_exit_at or "",
            "total_tokens": t.total_tokens,
            "total_cost": f"{t.total_cost:.6f}",
            "timing_sec": f"{t.timing_sec:.3f}",
        })

    def flush(self):
        if not self.rows: return
        fp = self.out / "per_sample.csv"
        with fp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader(); w.writerows(self.rows)

def write_summary(out_dir: str, summary: dict):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

# =========================
# NEW: Per-Stage CSV Logger
# =========================

class CsvStageLogger:
    """
    Writes a per-stage CSV with one row per (stage, subcall).
    For APO, splits 'helper' and 'target'. For others, a single row.
    Pulls ScaleDown usage fields if present under model_usage[...]['meta']['scaledown_usage'].
    """
    def __init__(self, out_dir: str):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []

    def _sd_fields(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        sd = (meta or {}).get("scaledown_usage") or {}
        full = sd.get("full_usage") or {}
        comp = sd.get("compressed_usage") or {}
        cmpn = sd.get("comparison") or {}
        return {
            "sd_full_tokens": full.get("tokens"),
            "sd_full_cost": full.get("cost"),
            "sd_compressed_tokens": comp.get("tokens"),
            "sd_compressed_cost": comp.get("cost"),
            "sd_savings_tokens": cmpn.get("tokens") or cmpn.get("savings"),
            "sd_savings_cost": cmpn.get("cost"),
        }

    def _push_row(self, base: Dict[str, Any], usage: Dict[str, Any], sub: str = ""):
        meta = usage.get("meta") or {}
        row = {
            **base,
            "sub": sub,  # "", "helper", "target"
            "model": usage.get("model", ""),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost"),
            **self._sd_fields(meta),
        }
        self.rows.append(row)

    def log_trace(self, t) -> None:
        for res in t.stage_results:
            stage_type = (res.evidence or {}).get("stage_type", "")
            base = {
                "qid": t.qid,
                "stage_id": res.stage_id,
                "stage_type": stage_type,
                "should_exit": bool(res.should_exit),
                "verdict": (res.evidence or {}).get("verdict", ""),
                "p": (res.evidence or {}).get("p", ""),
                "answer_snip": (res.answer or "")[:200],
            }
            usage = res.model_usage or {}
            if "prompt_tokens" in usage:
                self._push_row(base, usage)
            else:
                for sub, sub_usage in usage.items():
                    if isinstance(sub_usage, dict):
                        self._push_row(base, sub_usage, sub=sub)

    def flush(self):
        if not self.rows: return
        fp = self.out / "per_stage.csv"
        with fp.open("w", newline="", encoding="utf-8") as f:
            cols = [
                "qid","stage_id","stage_type","sub","model",
                "should_exit","verdict","p","answer_snip",
                "prompt_tokens","completion_tokens","cost",
                "sd_full_tokens","sd_full_cost",
                "sd_compressed_tokens","sd_compressed_cost",
                "sd_savings_tokens","sd_savings_cost",
            ]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(self.rows)
