"""Integration tests for FastAPI /rag endpoint schema."""

from llm_engineering.infrastructure.inference_pipeline_api import QueryRequest, QueryResponse, app


class TestFastAPISchema:
    def test_app_creates(self):
        assert app is not None
        assert app.title == "FastAPI"

    def test_rag_endpoint_exists(self):
        routes = [r.path for r in app.routes]
        assert "/rag" in routes

    def test_rag_accepts_post(self):
        rag_route = next(r for r in app.routes if getattr(r, "path", None) == "/rag")
        assert "POST" in rag_route.methods

    def test_query_request_model(self):
        req = QueryRequest(query="test question")
        assert req.query == "test question"

    def test_query_response_model(self):
        resp = QueryResponse(answer="test answer")
        assert resp.answer == "test answer"

    def test_rag_endpoint_method_is_post(self):
        rag_route = next(r for r in app.routes if getattr(r, "path", None) == "/rag")
        assert "GET" not in rag_route.methods
