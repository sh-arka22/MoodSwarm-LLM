"""RAG evaluation — Recall@K, MRR baselines on a test set of query→expected-chunk pairs.

Each test case maps a query to chunk IDs that SHOULD appear in the top-K results.
Metrics:
  - Recall@K: fraction of expected chunks found in top-K results
  - MRR: mean reciprocal rank of the first relevant result

Usage:
    poetry run python -m tools.rag_eval              # Real mode
    poetry run python -m tools.rag_eval --mock       # Mock mode
    poetry run python -m tools.rag_eval --k 5        # Evaluate at K=5
"""

import time
from dataclasses import dataclass, field

import click

from llm_engineering.application.rag.retriever import ContextRetriever

# ---------------------------------------------------------------------------
# Test set: query → expected chunk ID prefixes (first 8 hex chars of UUID)
# These were identified by inspecting `tools/qdrant_inspect.py sample` output.
# ---------------------------------------------------------------------------

TEST_SET: list[dict] = [
    {
        "query": "How does the inference pipeline work in the LLM system?",
        "expected_ids": ["86795ff2", "0b0dd1bf", "cfd06e18"],
        "description": "Inference pipeline architecture and client interaction",
    },
    {
        "query": "What is the RAG optimization and evaluation approach?",
        "expected_ids": ["ca408355", "0bb48b5f"],
        "description": "Advanced RAG techniques + RAGAS evaluation",
    },
    {
        "query": "How does the data collection pipeline crawl information?",
        "expected_ids": ["d03a63c1", "834ebb30", "9270ca80"],
        "description": "Data pipeline, crawling from Medium/LinkedIn/GitHub",
    },
    {
        "query": "What is the LLM Twin concept and course structure?",
        "expected_ids": ["1a2d0e65", "61c835ba", "3b97e068"],
        "description": "LLM Twin definition, course overview, FTI architecture",
    },
    {
        "query": "How does fine-tuning with QLoRA work?",
        "expected_ids": ["1775682c", "607912c3", "5aa8c7f2"],
        "description": "QLoRA fine-tuning, Mistral, model registry",
    },
    {
        "query": "What role do vector databases play in storing embeddings?",
        "expected_ids": ["66102407", "806a8dce", "ca408355"],
        "description": "Qdrant storage, vector DB for RAG, embedding snapshots",
    },
    {
        "query": "How does CDC and streaming ingestion work in the feature pipeline?",
        "expected_ids": ["9270ca80", "9364f17f", "611131f7", "b4884877"],
        "description": "CDC pattern, Bytewax streaming, RabbitMQ",
    },
]


@dataclass
class EvalResult:
    query: str
    description: str
    expected_ids: list[str]
    retrieved_ids: list[str] = field(default_factory=list)
    recall: float = 0.0
    reciprocal_rank: float = 0.0
    latency_ms: float = 0.0


def evaluate_query(
    retriever: ContextRetriever,
    test_case: dict,
    k: int,
) -> EvalResult:
    query = test_case["query"]
    expected = set(test_case["expected_ids"])

    t0 = time.perf_counter()
    results = retriever.search(query, k=k, expand_to_n_queries=3)
    latency = (time.perf_counter() - t0) * 1000

    retrieved = [str(doc.id)[:8] for doc in results]

    # Recall@K: what fraction of expected chunks were retrieved?
    hits = expected & set(retrieved)
    recall = len(hits) / len(expected) if expected else 0.0

    # MRR: reciprocal rank of the first relevant result
    rr = 0.0
    for rank, rid in enumerate(retrieved, 1):
        if rid in expected:
            rr = 1.0 / rank
            break

    return EvalResult(
        query=query,
        description=test_case["description"],
        expected_ids=list(expected),
        retrieved_ids=retrieved,
        recall=recall,
        reciprocal_rank=rr,
        latency_ms=latency,
    )


def print_results(results: list[EvalResult], k: int, mock: bool) -> None:
    mode = "MOCK" if mock else "REAL"
    click.echo(f"\n{'='*80}")
    click.echo(f"RAG EVALUATION REPORT [{mode} mode, K={k}]")
    click.echo(f"{'='*80}")

    total_recall = 0.0
    total_mrr = 0.0
    total_latency = 0.0

    for i, r in enumerate(results, 1):
        hits = set(r.expected_ids) & set(r.retrieved_ids)
        click.echo(f"\n--- Test {i}: {r.description}")
        click.echo(f"    Query:     {r.query[:75]}{'...' if len(r.query) > 75 else ''}")
        click.echo(f"    Expected:  {r.expected_ids}")
        click.echo(f"    Retrieved: {r.retrieved_ids}")
        click.echo(f"    Hits:      {list(hits)} ({len(hits)}/{len(r.expected_ids)})")
        click.echo(f"    Recall@{k}: {r.recall:.3f}   RR: {r.reciprocal_rank:.3f}   Latency: {r.latency_ms:.0f}ms")

        total_recall += r.recall
        total_mrr += r.reciprocal_rank
        total_latency += r.latency_ms

    n = len(results)
    avg_recall = total_recall / n
    avg_mrr = total_mrr / n
    avg_latency = total_latency / n

    click.echo(f"\n{'='*80}")
    click.echo(f"AGGREGATE METRICS ({n} test cases)")
    click.echo(f"{'='*80}")
    click.echo(f"  Mean Recall@{k}:  {avg_recall:.3f}")
    click.echo(f"  MRR:             {avg_mrr:.3f}")
    click.echo(f"  Avg Latency:     {avg_latency:.0f}ms")
    click.echo(f"{'='*80}\n")


@click.command(help="Evaluate RAG retrieval with Recall@K and MRR on a curated test set.")
@click.option("--k", default=3, help="Number of results to retrieve per query (default: 3).")
@click.option("--mock", is_flag=True, default=False, help="Run in mock mode (no OpenAI API calls).")
def cli(k: int, mock: bool) -> None:
    retriever = ContextRetriever(mock=mock)

    results = []
    for test_case in TEST_SET:
        result = evaluate_query(retriever, test_case, k=k)
        results.append(result)

    print_results(results, k=k, mock=mock)


if __name__ == "__main__":
    cli()
