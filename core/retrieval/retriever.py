class Retriever:
    def __init__(self, index, embedder):
        self.index = index
        self.embedder = embedder

    def retrieve(self, task: str, k=3):
        embedding = self.embedder.embed(task)
        results = self.index.search(embedding, k=k)
        return results
