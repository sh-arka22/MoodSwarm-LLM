"""Semantic search smoke test — embed a query and search all embedded collections."""

import click
from loguru import logger

from llm_engineering.application.preprocessing import EmbeddingDispatcher
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from llm_engineering.domain.queries import EmbeddedQuery, Query

EMBEDDED_CLASSES = [EmbeddedPostChunk, EmbeddedArticleChunk, EmbeddedRepositoryChunk]


def search_all(query_text: str, k: int = 3) -> list:
    """Embed a query string and search across all embedded collections."""
    query = Query.from_str(query_text)
    embedded_query: EmbeddedQuery = EmbeddingDispatcher.dispatch(query)

    logger.info(
        f"Query embedded: dim={len(embedded_query.embedding)}, "
        f"content='{embedded_query.content[:80]}...'"
    )

    all_results = []
    for cls in EMBEDDED_CLASSES:
        try:
            results = cls.search(
                query_vector=embedded_query.embedding,
                limit=k,
            )
            if results:
                logger.info(f"  {cls.get_collection_name()}: {len(results)} hits")
                all_results.extend(results)
        except Exception as e:
            logger.warning(f"  {cls.get_collection_name()}: skipped ({e})")

    return all_results


@click.command(help="Run semantic search across all embedded collections.")
@click.option("--query", "-q", required=True, help="Search query text.")
@click.option("--k", default=3, help="Results per collection (default: 3).")
def cli(query: str, k: int):
    results = search_all(query, k=k)

    if not results:
        click.echo("\nNo results found across any collection.")
        return

    click.echo(f"\n{'='*70}")
    click.echo(f"Query: {query}")
    click.echo(f"Total results: {len(results)}")
    click.echo(f"{'='*70}\n")

    for i, doc in enumerate(results, 1):
        collection = doc.get_collection_name()
        author = getattr(doc, "author_full_name", "unknown")
        content = doc.content
        if len(content) > 300:
            content = content[:300] + "..."

        click.echo(f"[{i}] {collection} | author={author}")
        click.echo(f"    {content}")
        click.echo()


if __name__ == "__main__":
    cli()
