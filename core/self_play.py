import random
from eval.datasets import load_tasks


class SelfPlayEngine:
    """
    Generates synthetic training episodes to improve routing.
    """

    def __init__(self, router):
        self.router = router
        self.tasks = load_tasks()

    def run_episode(self, steps: int = 10):
        for _ in range(steps):
            task = random.choice(self.tasks)
            features = self.router.extract_features(task.prompt)
            strategy = self.router.select_strategy(features)

            result = strategy.execute_sync(task.prompt)
            self.router.update(result)

    def run_background(self, rounds: int = 5):
        print("🧪 Running synthetic self-play...")
        for i in range(rounds):
            self.run_episode(steps=20)
            print(f"  Round {i+1} complete")
