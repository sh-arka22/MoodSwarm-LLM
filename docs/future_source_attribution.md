# Source Attribution in RAG Responses

## Problem

The RAG pipeline retrieves chunks from Qdrant that carry rich metadata — `link`, `platform`, `author_full_name` — but this metadata is discarded in `to_context()` before the response reaches the user. The final API response contains only the generated answer text with no indication of where the information came from.

This means users cannot:
- Verify claims against original sources
- Read the full article/repo behind a chunk
- Understand which platforms contributed to the answer

## Current Flow (metadata loss)

```
Qdrant chunk (has link, platform, author_full_name)
    → EmbeddedChunk.to_context()   # returns only text content
    → InferenceExecutor              # prompt built from text only
    → FastAPI /rag response          # {"answer": "..."}
```

## Proposed Solution

### 1. Add `Source` model and `to_sources()` method — `llm_engineering/domain/embedded_chunks.py`

```python
from pydantic import BaseModel

class Source(BaseModel):
    title: str | None = None
    url: str | None = None
    platform: str | None = None
    author: str | None = None

class EmbeddedChunk(VectorBaseDocument):
    # ... existing fields ...

    def to_sources(self) -> Source:
        """Extract source attribution metadata from chunk."""
        return Source(
            title=getattr(self, "title", None),
            url=getattr(self, "link", None),
            platform=getattr(self, "platform", None),
            author=getattr(self, "author_full_name", None),
        )
```

### 2. Return sources in API response — `llm_engineering/infrastructure/inference_pipeline_api.py`

```python
@app.post("/rag")
async def rag(request: RagRequest):
    # ... existing retrieval + inference ...

    # Deduplicate sources by URL
    seen_urls = set()
    sources = []
    for chunk in retrieved_chunks:
        source = chunk.to_sources()
        if source.url and source.url not in seen_urls:
            seen_urls.add(source.url)
            sources.append(source.model_dump())

    return {
        "answer": answer,
        "sources": sources,
    }
```

### 3. Show sources in CLI — `tools/rag.py`

```python
# After printing the answer
if sources:
    print("\n--- Sources ---")
    for i, src in enumerate(sources, 1):
        parts = [src.get("url", "")]
        if src.get("platform"):
            parts.append(f"({src['platform']})")
        print(f"  [{i}] {' '.join(parts)}")
```

## Expected API Response

```json
{
  "answer": "RAG systems combine retrieval with generation by...",
  "sources": [
    {
      "title": "Building RAG Systems",
      "url": "https://medium.com/@author/building-rag-systems",
      "platform": "medium",
      "author": "Arkajyoti Saha"
    },
    {
      "title": null,
      "url": "https://github.com/user/rag-example",
      "platform": "github",
      "author": "Arkajyoti Saha"
    }
  ]
}
```

## Files Changed

| File | Change |
|------|--------|
| `llm_engineering/domain/embedded_chunks.py` | Add `Source` model + `to_sources()` method |
| `llm_engineering/infrastructure/inference_pipeline_api.py` | Return `sources[]` in `/rag` response |
| `tools/rag.py` | Display source URLs in CLI output |

## Notes

- Source deduplication is by URL to avoid repeating the same article when multiple chunks come from it.
- The `Source` model fields are all optional since not every chunk type has every metadata field.
- No changes to Qdrant schema or the retrieval pipeline itself — metadata already exists in stored chunks.
