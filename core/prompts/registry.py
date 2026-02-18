from core.prompts.default import DefaultTemplate


class PromptRegistry:
    def __init__(self):
        self.templates = {
            "default": DefaultTemplate(),
        }

    def get(self, name: str):
        return self.templates.get(name, self.templates["default"])
