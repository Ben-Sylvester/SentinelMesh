import json
import requests
import time

API_URL = "http://localhost:8000/run"
API_KEY = "benchmark-key"


def load_tasks(path):
    with open(path) as f:
        return json.load(f)


def run_task(prompt):
    r = requests.post(
        API_URL,
        headers={"x-api-key": API_KEY},
        json={"prompt": prompt},
        timeout=60
    )
    r.raise_for_status()
    return r.json()


def run_suite(task_file, rounds=5, sleep=1):
    suite = load_tasks(task_file)
    results = []

    for r in range(rounds):
        for case in suite["cases"]:
            response = run_task(case["prompt"])
            results.append({
                "task": suite["name"],
                "case_id": case["id"],
                "round": r,
                "output": response["output"],
                "trace": response["trace"],
            })
            time.sleep(sleep)

    return results
