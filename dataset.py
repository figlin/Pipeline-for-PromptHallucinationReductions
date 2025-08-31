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
    bleurt: float = 0.0
    llm_judge: float = 0.0
    mc_accuracy: float = 0.0

class StageAverages(BaseModel):
    average_results: Dict[str, EvalResult]

    def __str__(self):
        return self.model_dump_json(indent=2)

class AggregatedMetrics(EvalResult):
    def __init__(self):
        super().__init__()
        self._results = {}

    def add_stage_result(self, stage_name, result: EvalResult):
        if stage_name not in self._results:
            self._results[stage_name] = EvalResult()
        current = self._results[stage_name]
        for field in result.model_fields:
            setattr(current, field, getattr(current, field) + getattr(result, field))

    def get_stage_average(self, stage) -> EvalResult:
        attr_name = f"{stage.__class__.__name__}_result"
        values = self._results.get(attr_name, [])
        return self._compute_average(values)

    def all_stage_averages(self):
        return StageAverages(
            average_results = {
                stage_name: self._compute_average(vals)
                for stage_name, vals in self._results.items() if vals
                }
        )

    def _compute_average(self, values) -> EvalResult | None:
        if values.n == 0:
            return None
        return EvalResult(
                n=values.n,
                em=values.em / values.n,
                f1=values.f1 / values.n,
                rouge1=values.rouge1 / values.n,
                bleurt=values.bleurt / values.n,
                llm_judge=values.llm_judge / values.n,
                mc_accuracy=values.mc_accuracy / values.n,
            )


class Dataset:
    def __init__(self, source: Union[str, List[Dict[str, Any]]], range_filter: Optional[str] = None):
        self.examples: List[Example] = []
        self.schema: Optional[str] = None

        if isinstance(source, list):
            raw_data = source
        elif isinstance(source, str):
            raw_data = self._load_from_file(source)
        else:
            raise ValueError("Source must be a list of dicts or a path string")

        self._load(raw_data, range_filter)

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

    def _load(self, raw_data: List[Dict[str, Any]], range_filter: Optional[str] = None):
        if not raw_data:
            return

        # Apply range filter if provided
        if range_filter:
            if range_filter is str:
                try:
                    start, end = map(int, range_filter.split('-'))
                    # Convert to 0-based indexing
                    start_idx = max(0, start - 1)
                    end_idx = min(len(raw_data), end)
                    raw_data = raw_data[start_idx:end_idx]
                except (ValueError, AttributeError):
                    print(f"Warning: Invalid DATASET_RANGE format '{range_filter}'. Expected format: 'start-end' (e.g., '1-50'). Using full dataset.")
            
            elif isinstance(range_filter, int) and range_filter > 0:
                raw_data = raw_data[:range_filter]

        keys = set(raw_data[0].keys()) if raw_data else set()
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


def load_from_csv(csv_path: str, range_filter: Optional[str] = None) -> Dataset:
    """Load dataset from a CSV file."""
    return Dataset(csv_path, range_filter)


# Create a default dataset instance for internal use
# You can modify this to load a specific internal dataset
dataset = Dataset([])
