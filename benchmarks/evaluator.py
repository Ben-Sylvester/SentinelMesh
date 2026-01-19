def evaluate_case(case, result):
    score = 0.0
    text = result["output"].lower()

    if "expected_contains" in case:
        score += any(x.lower() in text for x in case["expected_contains"])

    if "expected_keywords" in case:
        matches = sum(k.lower() in text for k in case["expected_keywords"])
        score += matches / len(case["expected_keywords"])

    return score
