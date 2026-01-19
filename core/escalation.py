# 1) Router selects strategy (bandit)
# 2) Strategy executes
# 3) If disagreement > threshold:
#       → escalate to stronger strategy
#       → re-run
# 4) Return final result + explain trace
# 5) Bandit learns from final outcome



DISAGREEMENT_THRESHOLD = 0.35


def should_escalate(result) -> bool:
    """
    Decide whether a strategy result should trigger escalation.
    """
    raw = result.raw_outputs or {}

    disagreement = raw.get("disagreement")

    if disagreement is None:
        return False

    return disagreement >= DISAGREEMENT_THRESHOLD
