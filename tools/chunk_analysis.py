"""Chunk analysis CLI — analyze chunk sizes and token distributions per Qdrant collection."""

import click

from llm_engineering.infrastructure.db.qdrant import connection


def _scroll_all_points(collection_name: str) -> list:
    """Scroll through all points in a collection."""
    all_records = []
    offset = None
    while True:
        records, next_offset = connection.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        all_records.extend(records)
        if next_offset is None:
            break
        offset = next_offset
    return all_records


@click.command(help="Analyze chunk statistics for embedded_* Qdrant collections.")
@click.option("--collection", default=None, help="Specific collection to analyze. Defaults to all embedded_* collections.")
@click.option("--max-tokens", default=256, help="Max allowed tokens per chunk (embedding model limit).")
def main(collection: str | None, max_tokens: int):
    from llm_engineering.application.networks.embeddings import EmbeddingModelSingleton

    model = EmbeddingModelSingleton()
    tokenizer = model._model.tokenizer

    collections = connection.get_collections().collections
    target_names = [c.name for c in collections if c.name.startswith("embedded_")]

    if collection:
        if collection not in [c.name for c in collections]:
            click.echo(f"Collection '{collection}' not found.")
            return
        target_names = [collection]

    if not target_names:
        click.echo("No embedded_* collections found.")
        return

    all_pass = True

    for col_name in sorted(target_names):
        records = _scroll_all_points(col_name)
        if not records:
            click.echo(f"\n{col_name}: empty")
            continue

        char_lengths = []
        token_counts = []
        over_limit = []

        for record in records:
            content = (record.payload or {}).get("content", "")
            char_len = len(content)
            tokens = tokenizer.encode(content, add_special_tokens=False)
            tok_count = len(tokens)

            char_lengths.append(char_len)
            token_counts.append(tok_count)

            if tok_count > max_tokens:
                over_limit.append((record.id, tok_count, content[:80]))

        click.echo(f"\n{'='*60}")
        click.echo(f"Collection: {col_name}  ({len(records)} chunks)")
        click.echo(f"{'='*60}")
        click.echo(f"  Characters:  min={min(char_lengths):>6}  max={max(char_lengths):>6}  avg={sum(char_lengths)/len(char_lengths):>8.1f}")
        click.echo(f"  Tokens:      min={min(token_counts):>6}  max={max(token_counts):>6}  avg={sum(token_counts)/len(token_counts):>8.1f}")

        if over_limit:
            all_pass = False
            click.echo(f"\n  FAIL: {len(over_limit)} chunks exceed {max_tokens} token limit:")
            for point_id, tok, preview in over_limit[:5]:
                click.echo(f"    id={point_id} tokens={tok} content={preview}...")
        else:
            click.echo(f"  Token limit check ({max_tokens}): PASS")

    click.echo(f"\n{'='*60}")
    click.echo(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    click.echo(f"{'='*60}")


if __name__ == "__main__":
    main()
