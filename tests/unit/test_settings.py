"""Unit tests for application settings."""

from llm_engineering.settings import Settings


class TestSettings:
    def test_defaults_load(self):
        s = Settings(_env_file=None)
        assert s.DATABASE_NAME == "twin"
        assert s.QDRANT_DATABASE_PORT == 6333

    def test_openai_model_default(self):
        s = Settings(_env_file=None)
        assert s.OPENAI_MODEL_ID == "gpt-4o-mini"

    def test_embedding_model_default(self):
        s = Settings(_env_file=None)
        assert s.TEXT_EMBEDDING_MODEL_ID == "sentence-transformers/all-MiniLM-L6-v2"

    def test_inference_temperature_is_float(self):
        s = Settings(_env_file=None)
        assert isinstance(s.TEMPERATURE_INFERENCE, float)

    def test_max_new_tokens_default(self):
        s = Settings(_env_file=None)
        assert s.MAX_NEW_TOKENS_INFERENCE == 150
