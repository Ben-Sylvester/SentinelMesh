
#This is where learning validation happens
def extract_metrics(results):
    metrics = {
        "avg_reward": [],
        "cost": [],
        "strategy": []
    }

    for r in results:
        trace = r["trace"]
        metrics["avg_reward"].append(trace["reward"])
        metrics["strategy"].append(trace["strategy"])
        metrics["cost"].append(trace["cost_usd"])

    return metrics
