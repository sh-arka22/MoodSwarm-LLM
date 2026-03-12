from __future__ import annotations

from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings


class InferenceExecutor:
    def __init__(
        self,
        llm: Inference,
        query: str,
        context: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.query = query
        self.context = context if context else ""

        if prompt is None:
            self.prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}

### Response:
"""
        else:
            self.prompt = prompt

    def execute(self) -> str:
        input_text = self.prompt.format(query=self.query, context=self.context)
        self.llm.set_payload(
            inputs=input_text,
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE,
                "return_full_text": False,
            },
        )
        answer = self.llm.inference()[0]["generated_text"]

        if not answer and hasattr(self.llm, "payload"):
            self.llm.set_payload(
                inputs=input_text,
                parameters={
                    "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                    "repetition_penalty": 1.1,
                    "temperature": settings.TEMPERATURE_INFERENCE,
                    "return_full_text": True,
                },
            )
            full_text = self.llm.inference()[0]["generated_text"]
            answer = full_text[len(input_text) :]

        return answer.strip()
