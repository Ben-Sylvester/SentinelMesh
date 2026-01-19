def detect_regression(previous, current):
    regressions = []

    if current["avg_reward"] < previous["avg_reward"] * 0.9:
        regressions.append("Reward dropped")

    if current["cost"] > previous["cost"] * 1.2:
        regressions.append("Cost increased")

    return regressions
