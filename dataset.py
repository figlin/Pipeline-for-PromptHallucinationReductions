import json
import csv
from typing import Any, List, Optional, Union, Dict
from pathlib import Path

from pydantic import BaseModel
from datasets import load_dataset

from utils import normalize_text, safe_list, SCHEMA_REGISTRY

class QAResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None

class Example (BaseModel):
    qid: str
    question: str
    y_true: str  # best answer
    q_type: Optional[str] = None
    category: Optional[str] = None
    correct_answers: Optional[List[str]] = None
    incorrect_answers: Optional[List[str]] = None

class EvalResult (BaseModel):
    n: int = 0
    em: float = 0.0
    f1: float = 0.0
    rouge1: float = 0.0
    bleu1: float = 0.0
    llm_judge: float = 0.0
    mc_accuracy: float = 0.0

class Dataset:
    def __init__(self, source: Union[str, List[Dict[str, Any]]]):
        self.examples: List[Example] = []
        self.schema: Optional[str] = None

        if isinstance(source, list):
            raw_data = source
        elif isinstance(source, str):
            raw_data = self._load_from_file(source)
        else:
            raise ValueError("Source must be a list of dicts or a path string")

        self._load(raw_data)

    def _load_from_file(self, path: str) -> List[Dict[str, Any]]:
        path_obj = Path(path)

        if path_obj.suffix == ".jsonl":
            with open(path_obj, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        elif path_obj.suffix == ".csv":
            with open(path_obj, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return list(reader)
        else:
            ds_dict = load_dataset(path)
            first_split = list(ds_dict.keys())[0]
            ds = ds_dict[first_split]
            return ds.to_list()

    def _load(self, raw_data: List[Dict[str, Any]]):
        if not raw_data:
            return

        keys = set(raw_data[0].keys())
        for name, schema in SCHEMA_REGISTRY.items():
            if schema["keys"] <= keys:
                self.schema = name
                getattr(self, schema["parser"])(raw_data)
                return
        raise ValueError(f"Unknown dataset schema: {keys}")

    def _parse_simpleqa(self, raw_data: List[Dict[str, Any]]):
        for idx, row in enumerate(raw_data, start=1):
            ex = Example(
                qid=f"{self.schema}-{idx}",
                question=normalize_text(row.get("problem", "")),
                y_true=normalize_text(row.get("answer", "")),
                correct_answers=[normalize_text(a) for a in row.get("answers", [row.get("answer", "")])],
            )
            self.examples.append(ex)

    def _parse_truthfulqa(self, raw_data: List[Dict[str, Any]]):
        for idx, row in enumerate(raw_data, start=1):
            ex = Example(
                qid=f"{self.schema}-{idx}",
                question=normalize_text(row.get("Question", "")),
                y_true=normalize_text(row.get("Best Answer", "")),
                q_type=row.get("Type"),
                category=row.get("Category"),
                correct_answers=[normalize_text(ans) for ans in safe_list(row.get("Correct Answers"))],
                incorrect_answers=[normalize_text(ans) for ans in safe_list(row.get("Incorrect Answers"))],
            )
            self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]
