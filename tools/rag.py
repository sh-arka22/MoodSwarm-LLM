"""RAG retrieval pipeline CLI — end-to-end test of the full retrieval chain.

Pipeline: Query → SelfQuery → QueryExpansion → parallel vector search → dedup → Rerank → results

Usage:
    poetry run python -m tools.rag -q "How do RAG systems work?" --k 3
    poetry run python -m tools.rag -q "My name is Arkajyoti Saha. Write about LLMs." --k 9
    poetry run python -m tools.rag -q "What are best practices for ML deployment?" --mock
"""

import click
from loguru import logger

from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.domain.embedded_chunks import EmbeddedChunk


def format_results(query: str, documents: list[EmbeddedChunk], mock: bool) -> None:
    mode = "MOCK" if mock else "REAL"
    click.echo(f"\n{'='*70}")
    click.echo(f"RAG Retrieval [{mode} mode]")
    click.echo(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    click.echo(f"Results: {len(documents)}")
    click.echo(f"{'='*70}\n")

    if not documents:
        click.echo("No documents retrieved.")
        return

    for rank, doc in enumerate(documents, 1):
        collection = doc.get_collection_name()
        author = getattr(doc, "author_full_name", "unknown")
        content = doc.content
        if len(content) > 300:
            content = content[:300] + "..."

        click.echo(f"[{rank}] {collection} | author={author}")
        click.echo(f"    {content}")
        click.echo()


@click.command(help="Run the full RAG retrieval pipeline (SelfQuery → Expand → Search → Rerank).")
@click.option("--query", "-q", required=True, help="Search query text.")
@click.option("--k", default=3, help="Number of final results after reranking (default: 3).")
@click.option("--expand-to-n", default=3, help="Number of query variants for expansion (default: 3).")
@click.option("--mock", is_flag=True, default=False, help="Run in mock mode (no OpenAI API calls).")
def cli(query: str, k: int, expand_to_n: int, mock: bool) -> None:
    retriever = ContextRetriever(mock=mock)

    logger.info(f"Starting RAG retrieval (mock={mock}, k={k}, expand_to_n={expand_to_n})")
    documents = retriever.search(query, k=k, expand_to_n_queries=expand_to_n)

    format_results(query, documents, mock)


if __name__ == "__main__":
    cli()
