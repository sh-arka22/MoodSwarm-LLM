"""Unit tests for domain models."""

from llm_engineering.domain.queries import Query
from llm_engineering.domain.types import DataCategory


class TestDataCategory:
    def test_has_expected_categories(self):
        expected = {"POSTS", "ARTICLES", "REPOSITORIES", "QUERIES", "PROMPT"}
        actual = {c.name for c in DataCategory}
        assert expected.issubset(actual)

    def test_values_are_lowercase(self):
        for category in DataCategory:
            assert category.value == category.value.lower()

    def test_posts_value(self):
        assert DataCategory.POSTS == "posts"

    def test_articles_value(self):
        assert DataCategory.ARTICLES == "articles"


class TestQuery:
    def test_from_str_creates_query(self, sample_query_text):
        query = Query.from_str(sample_query_text)
        assert query.content == sample_query_text
        assert query.author_id is None
        assert query.author_full_name is None

    def test_from_str_strips_whitespace(self):
        query = Query.from_str("  \n  How does LLM work?  \n  ")
        assert query.content == "How does LLM work?"

    def test_replace_content_preserves_id(self):
        original = Query.from_str("original query")
        replaced = original.replace_content("new query")
        assert replaced.id == original.id
        assert replaced.content == "new query"

    def test_replace_content_preserves_metadata(self):
        original = Query.from_str("test")
        original.metadata = {"key": "value"}
        replaced = original.replace_content("new")
        assert replaced.metadata == {"key": "value"}

    def test_query_config_category(self):
        assert Query.Config.category == DataCategory.QUERIES
