from dataclasses import dataclass
from typing import List, Optional
import csv
import sys

from utils import Example

dataset: List[Example] = [
    Example(qid="1", question="What is the capital of France?", y_true="Paris"),
    Example(qid="2", question="Who wrote the play 'Romeo and Juliet'?", y_true="William Shakespeare"),
    Example(qid="3", question="What is the tallest mountain in the world above sea level?", y_true="Mount Everest"),
    Example(qid="4", question="In what year did the Apollo 11 mission land humans on the Moon?", y_true="1969"),
    Example(qid="5", question="What is the chemical symbol for gold?", y_true="Au"),
    Example(qid="6", question="What is the smallest prime number?", y_true="2"),
    Example(qid="7", question="What is the capital of Australia?", y_true="Canberra"),
    Example(qid="8", question="Who is the current UN Secretary-General?", y_true="António Guterres"),
]

def load_from_csv(filepath: str) -> List[Example]:
    """Loads a dataset from a CSV file, trying common column names."""
    examples = []
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            # Make column name detection case-insensitive
            reader.fieldnames = [name.lower() for name in reader.fieldnames or []]
            
            # Find column names for question, answer, and qid
            question_col = next((c for c in ["question", "prompt", "problem"] if c in reader.fieldnames), None)
            answer_col = next((c for c in ["answer", "y_true"] if c in reader.fieldnames), None)
            qid_col = next((c for c in ["qid", "id"] if c in reader.fieldnames), None)

            if not question_col:
                print(f"Error: CSV file '{filepath}' must have a 'question' or 'prompt' column.", file=sys.stderr)
                sys.exit(1)

            for i, row in enumerate(reader):
                # Normalize row keys to lower case
                row = {k.lower(): v for k, v in row.items()}
                examples.append(Example(
                    qid=row.get(qid_col, str(i + 1)) if qid_col else str(i + 1),
                    question=row[question_col],
                    y_true=row.get(answer_col) if answer_col else None
                ))
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{filepath}'", file=sys.stderr)
        sys.exit(1)
        
    return examples

