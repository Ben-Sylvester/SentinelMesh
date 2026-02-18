from core.prompts.base import PromptTemplate
from typing import Optional  # Import Optional for Python versions before 3.10


class DefaultTemplate(PromptTemplate):
    name = "default"

    def format(self, task: str, context: Optional[str] = None) -> str:  # Use Optional[str]
        if context:
            return f"""
            You are SentinelMesh.

            Context:
            {context}

            Task:
            {task}

            Provide a clear and structured answer.
            """
        return f"""
        You are SentinelMesh.

        Task:
        {task}

        Provide a clear and structured answer.
        """
