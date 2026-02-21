"""Qdrant inspection CLI — list, stats, sample, and search collections."""

import click
from loguru import logger

from llm_engineering.infrastructure.db.qdrant import connection


@click.group(help="Inspect Qdrant vector database collections.")
def cli():
    pass


@cli.command("list-collections")
def list_collections():
    """List all Qdrant collections with point counts."""
    collections = connection.get_collections().collections
    if not collections:
        click.echo("No collections found.")
        return

    click.echo(f"\n{'Collection':<30} {'Points':>10} {'Idx Vectors':>12} {'Status':<10}")
    click.echo("-" * 67)
    for col in sorted(collections, key=lambda c: c.name):
        info = connection.get_collection(col.name)
        click.echo(
            f"{col.name:<30} {info.points_count:>10} {info.indexed_vectors_count:>12} {info.status.value:<10}"
        )
    click.echo()


@cli.command("collection-stats")
@click.argument("name")
def collection_stats(name: str):
    """Show detailed stats for a collection."""
    try:
        info = connection.get_collection(name)
    except Exception:
        click.echo(f"Collection '{name}' not found.")
        return

    click.echo(f"\nCollection: {name}")
    click.echo(f"  Status:        {info.status}")
    click.echo(f"  Points count:  {info.points_count}")
    click.echo(f"  Indexed vecs:  {info.indexed_vectors_count}")
    click.echo(f"  Segments:      {info.segments_count}")

    vec_cfg = info.config.params.vectors
    if isinstance(vec_cfg, dict):
        if vec_cfg:
            for vec_name, params in vec_cfg.items():
                click.echo(f"  Vector '{vec_name}': size={params.size}, distance={params.distance}")
        else:
            click.echo("  Vectors:       None (payload-only collection)")
    else:
        click.echo(f"  Vector size:   {vec_cfg.size}")
        click.echo(f"  Distance:      {vec_cfg.distance}")
    click.echo()


@cli.command("sample")
@click.argument("name")
@click.option("--limit", default=3, help="Number of points to sample.")
@click.option("--with-vectors", is_flag=True, default=False, help="Include vectors in output.")
def sample(name: str, limit: int, with_vectors: bool):
    """Sample points from a collection."""
    try:
        records, _next = connection.scroll(
            collection_name=name,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )
    except Exception:
        click.echo(f"Collection '{name}' not found or empty.")
        return

    if not records:
        click.echo(f"No points in '{name}'.")
        return

    click.echo(f"\nSampling {len(records)} points from '{name}':\n")
    for i, record in enumerate(records, 1):
        click.echo(f"--- Point {i} (id: {record.id}) ---")
        payload = record.payload or {}
        for key, value in payload.items():
            display = str(value)
            if len(display) > 200:
                display = display[:200] + "..."
            click.echo(f"  {key}: {display}")
        if with_vectors and record.vector:
            vec = record.vector
            if isinstance(vec, list) and len(vec) > 6:
                vec_str = f"[{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}, ..., {vec[-1]:.4f}] (dim={len(vec)})"
            else:
                vec_str = str(vec)
            click.echo(f"  vector: {vec_str}")
        click.echo()


@cli.command("search")
@click.argument("name")
@click.option("--query", required=True, help="Search query text.")
@click.option("--limit", default=5, help="Number of results.")
def search(name: str, query: str, limit: int):
    """Semantic search in an embedded collection."""
    from llm_engineering.application.networks.embeddings import EmbeddingModelSingleton

    model = EmbeddingModelSingleton()

    # Verify collection has vectors
    try:
        info = connection.get_collection(name)
    except Exception:
        click.echo(f"Collection '{name}' not found.")
        return

    vec_cfg = info.config.params.vectors
    has_vectors = not isinstance(vec_cfg, dict) or bool(vec_cfg)
    if not has_vectors:
        click.echo(f"Collection '{name}' has no vectors. Use an embedded_* collection.")
        return

    query_vector = model(query, to_list=True)

    response = connection.query_points(
        collection_name=name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    if not response.points:
        click.echo("No results found.")
        return

    click.echo(f"\nSearch results for '{query}' in '{name}' (top {limit}):\n")
    for i, hit in enumerate(response.points, 1):
        payload = hit.payload or {}
        content = payload.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        author = payload.get("author_full_name", "unknown")
        click.echo(f"  [{i}] score={hit.score:.4f} | author={author}")
        click.echo(f"      {content}")
        click.echo()


if __name__ == "__main__":
    cli()
