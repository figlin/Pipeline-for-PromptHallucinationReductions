import csv, json
from pathlib import Path
from typing import List

class CsvLogger:
    def __init__(self, out_dir: str):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
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
