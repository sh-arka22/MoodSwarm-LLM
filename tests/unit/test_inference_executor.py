"""Unit tests for InferenceExecutor prompt formatting."""

from unittest.mock import MagicMock

from llm_engineering.model.inference.run import InferenceExecutor


class TestInferenceExecutorPrompt:
    def test_default_prompt_has_alpaca_markers(self):
        llm = MagicMock()
        executor = InferenceExecutor(llm=llm, query="test", context="ctx")
        assert "### Instruction:" in executor.prompt
        assert "### Response:" in executor.prompt

    def test_prompt_substitutes_query_and_context(self):
        llm = MagicMock()
        executor = InferenceExecutor(llm=llm, query="my query", context="my context")
        formatted = executor.prompt.format(query="my query", context="my context")
        assert "my query" in formatted
        assert "my context" in formatted

    def test_custom_prompt_overrides_default(self):
        llm = MagicMock()
        custom = "Custom: {query} with {context}"
        executor = InferenceExecutor(llm=llm, query="q", context="c", prompt=custom)
        assert executor.prompt == custom

    def test_empty_context_defaults_to_empty_string(self):
        llm = MagicMock()
        executor = InferenceExecutor(llm=llm, query="test")
        assert executor.context == ""

    def test_execute_calls_llm(self):
        llm = MagicMock()
        llm.inference.return_value = [{"generated_text": "answer"}]
        executor = InferenceExecutor(llm=llm, query="q", context="c")
        result = executor.execute()
        assert result == "answer"
        llm.set_payload.assert_called_once()
        llm.inference.assert_called_once()
