"""RAG parameter tuning — measure latency per stage and test parameter combinations.

Usage:
    poetry run python -m tools.rag_tuning
    poetry run python -m tools.rag_tuning --mock
"""

import time

import click

from llm_engineering.application.preprocessing.dispatchers import EmbeddingDispatcher
from llm_engineering.application.rag.query_expansion import QueryExpansion
from llm_engineering.application.rag.reranking import Reranker
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.rag.self_query import SelfQuery
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from llm_engineering.domain.queries import Query

EMBEDDED_CLASSES = [EmbeddedPostChunk, EmbeddedArticleChunk, EmbeddedRepositoryChunk]

TEST_QUERIES = [
    "How do RAG systems integrate with vector databases?",
    "What are best practices for LLM deployment in production?",
    "My name is Paul Iusztin. Write about fine-tuning techniques.",
]


def timed(fn, *args, **kwargs):
    """Run a function and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


def stage_latency_test(mock: bool) -> None:
    """Measure latency of each individual RAG stage."""
    click.echo(f"\n{'='*70}")
    click.echo("STAGE LATENCY TEST")
    click.echo(f"{'='*70}")

    query_text = TEST_QUERIES[0]
    query = Query.from_str(query_text)

    # Stage 1: SelfQuery
    sq = SelfQuery(mock=mock)
    enriched_query, ms = timed(sq.generate, query)
    click.echo(f"\n1. SelfQuery:        {ms:7.1f} ms  (author={enriched_query.author_full_name})")

    # Stage 2: QueryExpansion
    qe = QueryExpansion(mock=mock)
    expanded, ms = timed(qe.generate, query, expand_to_n=3)
    click.echo(f"2. QueryExpansion:   {ms:7.1f} ms  (n={len(expanded)} queries)")

    # Stage 3: Embedding
    embed_query, ms = timed(EmbeddingDispatcher.dispatch, query)
    click.echo(f"3. Embedding:        {ms:7.1f} ms  (dim={len(embed_query.embedding)})")

    # Stage 4: Vector search (single collection)
    results, ms = timed(EmbeddedArticleChunk.search, query_vector=embed_query.embedding, limit=3)
    click.echo(f"4. Vector search:    {ms:7.1f} ms  ({len(results)} results from embedded_articles)")

    # Stage 5: Reranking
    if results:
        reranker = Reranker(mock=mock)
        reranked, ms = timed(reranker.generate, query=query, chunks=results, keep_top_k=min(3, len(results)))
        click.echo(f"5. Reranking:        {ms:7.1f} ms  ({len(reranked)} reranked)")


def parameter_sweep(mock: bool) -> None:
    """Test different parameter combinations and measure total latency + result counts."""
    click.echo(f"\n{'='*70}")
    click.echo("PARAMETER SWEEP")
    click.echo(f"{'='*70}")

    k_values = [3, 6, 9]
    expand_values = [1, 3, 5]

    click.echo(f"\n{'Query':<55} {'k':>3} {'exp':>4} {'results':>8} {'ms':>8}")
    click.echo("-" * 85)

    for query_text in TEST_QUERIES:
        for k in k_values:
            for expand_to_n in expand_values:
                retriever = ContextRetriever(mock=mock)
                results, ms = timed(retriever.search, query_text, k=k, expand_to_n_queries=expand_to_n)
                label = query_text[:52] + "..." if len(query_text) > 55 else query_text
                click.echo(f"{label:<55} {k:>3} {expand_to_n:>4} {len(results):>8} {ms:>7.0f}ms")


def edge_case_test(mock: bool) -> None:
    """Test edge cases for robustness."""
    click.echo(f"\n{'='*70}")
    click.echo("EDGE CASE TESTS")
    click.echo(f"{'='*70}\n")

    retriever = ContextRetriever(mock=mock)
    cases = [
        ("Empty-ish query", "hello"),
        ("Very long query", "How do RAG systems work? " * 20),
        ("Non-English", "Wie funktionieren RAG-Systeme mit Vektordatenbanken?"),
        ("Author only", "My name is Paul Iusztin."),
        ("No context match", "Recipe for chocolate cake with almonds"),
    ]

    for label, query_text in cases:
        try:
            results, ms = timed(retriever.search, query_text, k=3, expand_to_n_queries=3)
            click.echo(f"  {label:<25} -> {len(results)} results  ({ms:.0f}ms)  PASS")
        except Exception as e:
            click.echo(f"  {label:<25} -> ERROR: {e}")


@click.command(help="RAG parameter tuning: stage latency, parameter sweep, edge cases.")
@click.option("--mock", is_flag=True, default=False, help="Run in mock mode (no OpenAI API calls).")
def cli(mock: bool) -> None:
    mode = "MOCK" if mock else "REAL"
    click.echo(f"\nRAG Tuning Suite [{mode} mode]")

    stage_latency_test(mock)
    parameter_sweep(mock)
    edge_case_test(mock)

    click.echo(f"\n{'='*70}")
    click.echo("TUNING COMPLETE")
    click.echo(f"{'='*70}\n")


if __name__ == "__main__":
    cli()
