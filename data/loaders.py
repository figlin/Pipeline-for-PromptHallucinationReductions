# data/loaders.py

import csv
from pathlib import Path
from typing import List
from utils import Example as UExample

DATA_DIR = Path(__file__).resolve().parent
_HAS_META = "meta" in getattr(UExample, "__annotations__", {})

def load_simpleqa(path: str = None) -> List[UExample]:
    p = Path(path) if path else DATA_DIR / "simple_qa_test_set.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing SimpleQA file at {p}. If PR #7 isn’t merged, drop the CSV into /data.")
    rows: List[UExample] = []
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r, start=1):
            question = (row.get("problem") or "").strip()
            y_true = (row.get("answer") or "").strip() or None
            meta = {"metadata": row.get("metadata", "")}
            if _HAS_META:
                rows.append(UExample(qid=str(i), question=question, y_true=y_true, meta=meta))
            else:
                rows.append(UExample(qid=str(i), question=question, y_true=y_true))
    return rows

def load_truthfulqa(path: str = None) -> List[UExample]:
    p = Path(path) if path else DATA_DIR / "TruthfulQA.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing TruthfulQA file at {p}. If PR #7 isn’t merged, drop the CSV into /data.")
    rows: List[UExample] = []
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r, start=1):
            question = (row.get("Question") or "").strip()
            gold = (row.get("Best Answer") or "").strip()
            if not gold:
                ca = (row.get("Correct Answers") or "").strip()
                if ca:
                    gold = ca.split(";")[0].split("|")[0].strip()
            meta = {
                "type": row.get("Type", ""),
                "category": row.get("Category", ""),
                "best_incorrect": row.get("Best Incorrect Answer", ""),
                "correct_answers_raw": row.get("Correct Answers", ""),
                "incorrect_answers_raw": row.get("Incorrect Answers", ""),
                "source": row.get("Source", "")
            }
            if _HAS_META:
                rows.append(UExample(qid=str(i), question=question, y_true=gold or None, meta=meta))
            else:
                rows.append(UExample(qid=str(i), question=question, y_true=gold or None))
    return rows
