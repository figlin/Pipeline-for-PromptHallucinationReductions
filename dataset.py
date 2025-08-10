from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Example:
    qid: str
    question: str
    y_true: Optional[str] = None

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
