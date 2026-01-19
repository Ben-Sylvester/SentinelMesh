from typing import Dict
from difflib import SequenceMatcher

# Disagreement Utilities
def text_similarity(a: str, b: str) -> float:
    """
    Returns similarity between two strings [0..1]
    """
    return SequenceMatcher(None, a, b).ratio()


def disagreement_score(outputs: Dict[str, str]) -> float:
    """
    0.0 = perfect agreement
    1.0 = complete disagreement
    """
    if len(outputs) < 2:
        return 0.0

    similarities = []
    values = list(outputs.values())

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            sim = text_similarity(values[i], values[j])
            similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities)
    return round(1.0 - avg_similarity, 3)
