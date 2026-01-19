from dataclasses import dataclass
from typing import List


@dataclass
class Task:
    id: str
    prompt: str
    expected_contains: str
    difficulty: str


def load_tasks() -> List[Task]:
    return [
        Task(
            id="math_1",
            prompt="What is 17 * 23?",
            expected_contains="391",
            difficulty="easy",
        ),
        Task(
            id="reasoning_1",
            prompt="If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
            expected_contains="yes",
            difficulty="medium",
        ),
        Task(
            id="code_1",
            prompt="Write a Python function that checks if a number is prime.",
            expected_contains="def",
            difficulty="medium",
        ),
        Task(
            id="analysis_1",
            prompt="Summarize why federated learning improves privacy.",
            expected_contains="privacy",
            difficulty="hard",
        ),
    ]
