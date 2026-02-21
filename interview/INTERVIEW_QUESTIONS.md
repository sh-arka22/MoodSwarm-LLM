# 🎓 MoodSwarm – Masters in ML Interview Questions

> These questions are derived directly from the **MoodSwarm LLM Twin & MLOps Platform** codebase.
> They target a candidate applying to a **Masters in Machine Learning** program and cover
> system design, NLP, vector databases, MLOps, software engineering patterns, and applied ML theory.
>
> Every question references **actual implemented code** in this repository.

---

## 1. System Architecture & FTI Design

**Q1. Explain the FTI (Feature, Training, Inference) architecture used in MoodSwarm. Why decompose into three separate pipelines?**

*What we built:* The repo defines distinct ZenML pipelines — `digital_data_etl.py` (extraction), `feature_engineering.py` (feature), and planned training/inference pipelines.

*Expected answer:*
- Feature pipeline transforms raw crawled data → cleaned documents → chunked + embedded vectors in Qdrant
- Training pipeline (future) will fine-tune models using datasets generated from the feature store
- Inference pipeline (future) will serve queries via RAG retrieval + LLM generation
- Separation enables independent versioning, scaling, and debugging of each stage

---

**Q2. The codebase uses Domain-Driven Design (DDD). Looking at the `llm_engineering/` package, explain the purpose of each layer: `domain/`, `application/`, `infrastructure/`, `model/`.**

*Actual structure:*
```
llm_engineering/
├── domain/          # Data models, types, exceptions (UserDocument, ArticleDocument, Chunk, etc.)
├── application/     # Business logic (crawlers/, preprocessing/, networks/)
├── infrastructure/  # External system connectors (db/mongo.py, db/qdrant.py)
└── model/           # ML model code (future: SFT, DPO)
```

*Expected answer:*
- `domain/` = pure data definitions with no external dependencies
- `application/` = use-case orchestration (crawling, cleaning, chunking, embedding)
- `infrastructure/` = adapters to external systems — dependencies point *inward* (infrastructure depends on domain, never the reverse)
- `model/` = ML-specific training and inference code
- This layering prevents tight coupling — e.g., swapping MongoDB for PostgreSQL only requires changing `infrastructure/db/`

---

**Q3. Why does MoodSwarm use both MongoDB and Qdrant? What tradeoffs does this dual-store architecture introduce?**

*Actual code:*
- `infrastructure/db/mongo.py` — singleton connector for raw document storage
- `infrastructure/db/qdrant.py` — singleton connector for vector embeddings
- `domain/documents.py` extends `NoSQLBaseDocument` (MongoDB ODM)
- `domain/cleaned_documents.py`, `chunks.py`, `embedded_chunks.py` extend `VectorBaseDocument` (Qdrant ODM)

*Expected answer:*
- MongoDB stores raw schemaless documents (articles, posts, repos) with flexible querying by author, link, etc.
- Qdrant stores dense vector embeddings optimized for approximate nearest neighbor (ANN) search
- Tradeoffs: operational complexity (two databases to manage), data consistency between stores, need for sync mechanisms (CDC)

---

## 2. Data Engineering & ETL Pipeline

**Q4. Walk through how a URL becomes a stored document. Reference the actual classes involved.**

*Actual code path:*
```
tools/run.py --run-etl  →  pipelines/digital_data_etl.py
  → steps/etl/get_or_create_user.py     (finds/creates UserDocument in MongoDB)
  → steps/etl/crawl_links.py            (dispatches URLs to crawlers)
    → CrawlerDispatcher.get_crawler(url)  (regex matching in dispatcher.py)
      → MediumCrawler / GithubCrawler / CustomArticleCrawler
        → Stores ArticleDocument / RepositoryDocument / PostDocument in MongoDB
```

*Expected answer should trace:* CLI trigger → ZenML pipeline → user lookup → dispatcher regex matching → specific crawler scraping → MongoDB document storage with deduplication check

---

**Q5. In `dispatcher.py`, the `CrawlerDispatcher` uses a builder pattern with `register()`. How does it decide which crawler handles a URL? What happens for unrecognized URLs?**

*Actual code (line 29-40):*
```python
def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
    parsed_domain = urlparse(domain)
    domain = parsed_domain.netloc
    self._crawlers[r"https://(www\.)?{}/*".format(re.escape(domain))] = crawler

def get_crawler(self, url: str) -> BaseCrawler:
    for pattern, crawler in self._crawlers.items():
        if re.match(pattern, url):
            return crawler()
    logger.warning(f"No crawler found for {url}. Defaulting to CustomArticleCrawler.")
    return CustomArticleCrawler()
```

*Expected answer:*
- `register()` stores a regex pattern → crawler class mapping using `urlparse` to extract the domain
- `get_crawler()` iterates through registered patterns, returning the first match
- Unmatched URLs fall back to `CustomArticleCrawler` with a warning — this ensures the pipeline never crashes on unknown sources

---

**Q6. How does the ETL handle deduplication and retry logic? Why use exponential backoff?**

*Actual implementation:*
- Deduplication: crawlers check `self.model.find(link=link)` — if document exists, skip
- Retry: `tenacity` decorator with 3 attempts, exponential backoff on `ConnectionError`, `TimeoutError`, `OSError`

*Expected answer:*
- Deduplication prevents duplicate documents on re-runs, maintaining data integrity
- Exponential backoff avoids thundering-herd effects — if a server is overloaded, fixed intervals maintain constant pressure, while exponential waits give recovery time
- Combined, these make the pipeline **idempotent** — safe to re-run without side effects

---

**Q7. The `query_data_warehouse` step uses `ThreadPoolExecutor` to fetch articles, posts, and repositories concurrently. Why? What risks does this introduce?**

*Actual code (lines 37-55):*
```python
def fetch_all_data(user: UserDocument) -> dict[str, list[NoSQLBaseDocument]]:
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(__fetch_articles, user_id): "articles",
            executor.submit(__fetch_posts, user_id): "posts",
            executor.submit(__fetch_repositories, user_id): "repositories",
        }
        results = {}
        for future in as_completed(future_to_query):
            ...
```

*Expected answer:*
- Concurrent fetching reduces total I/O wait time (3 sequential queries vs 1 parallel batch)
- Risks: connection pool exhaustion if MongoDB max connections are limited, error isolation (one query failure shouldn't kill others — handled via try/except per future), thread safety of shared state

---

## 3. Feature Pipeline — Clean → Chunk → Embed

**Q8. The `feature_engineering` pipeline runs 4 steps. Trace the exact data flow and explain why `load_to_vector_db` is called twice.**

*Actual code (`pipelines/feature_engineering.py`):*
```python
@pipeline
def feature_engineering(author_full_names: list[str]) -> list[str]:
    raw_documents = fe_steps.query_data_warehouse(author_full_names)
    cleaned_documents = fe_steps.clean_documents(raw_documents)
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)       # cleaned → Qdrant
    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)      # embedded → Qdrant
    return [last_step_1.invocation_id, last_step_2.invocation_id]
```

*Expected answer:*
- First `load_to_vector_db` stores **cleaned documents** in Qdrant (in `cleaned_posts`, `cleaned_articles`, `cleaned_repositories` collections — with `use_vector_index=False`)
- Second `load_to_vector_db` stores **embedded chunks** in Qdrant (in `embedded_posts`, `embedded_articles`, `embedded_repositories` collections — with vectors for similarity search)
- Cleaned docs are stored for full-text retrieval; embedded chunks enable semantic vector search — supporting hybrid retrieval strategies

---

**Q9. The `CleaningDispatcher` in `dispatchers.py` converts a `NoSQLBaseDocument` into a `VectorBaseDocument`. How does it determine the correct handler?**

*Actual code:*
```python
class CleaningDispatcher:
    factory = CleaningHandlerFactory()

    @classmethod
    def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument:
        data_category = DataCategory(data_model.get_collection_name())
        handler = cls.factory.create_handler(data_category)
        clean_model = handler.clean(data_model)
        ...
```

*Expected answer:*
- `data_model.get_collection_name()` returns the MongoDB collection name (e.g., `"articles"`)
- This string is cast to `DataCategory` enum (e.g., `DataCategory.ARTICLES`)
- `CleaningHandlerFactory.create_handler()` uses if/elif matching on the enum to return `ArticleCleaningHandler`
- The handler's `clean()` method returns a `CleanedArticleDocument` (a `VectorBaseDocument`)

---

**Q10. The `EmbeddingDispatcher` handles both single documents and batches. Looking at the actual code, explain the polymorphic dispatch logic and the assertion check.**

*Actual code (lines 104-131 of `dispatchers.py`):*
```python
@classmethod
def dispatch(cls, data_model: VectorBaseDocument | list[VectorBaseDocument]) -> ...:
    is_list = isinstance(data_model, list)
    if not is_list:
        data_model = [data_model]
    if len(data_model) == 0:
        return []
    data_category = data_model[0].get_category()
    assert all(dm.get_category() == data_category for dm in data_model), 
        "Data models must be of the same category."
    handler = cls.factory.create_handler(data_category)
    embedded_chunk_model = handler.embed_batch(data_model)
    ...
```

*Expected answer:*
- Accepts single or list input — normalizes to list internally, preserving the original format on output
- The `assert` ensures all documents in a batch are the same `DataCategory` — mixing posts with articles would produce incorrect embeddings
- This is important because each category may use a different embedding handler (even if currently all use the same model, the architecture supports per-type models)

---

**Q11. In the `chunk_and_embed` step (`rag.py`), chunks are processed in batches of 10 for embedding. Why batch instead of one-at-a-time or all-at-once?**

*Actual code:*
```python
for batched_chunks in utils.misc.batch(chunks, 10):
    batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
    embedded_chunks.extend(batched_embedded_chunks)
```

*Expected answer:*
- Batching enables efficient GPU/CPU utilization via `SentenceTransformer.encode()` batch processing
- One-at-a-time = high overhead per call; all-at-once = potential OOM for large documents
- Batch size 10 balances throughput vs memory usage
- The `batch()` utility in `utils/misc.py` yields chunks of a given size from a list

---

## 4. Domain Modeling & ODM Patterns

**Q12. Compare `NoSQLBaseDocument` (MongoDB) and `VectorBaseDocument` (Qdrant). How do their serialization methods differ?**

*Actual methods:*
- `NoSQLBaseDocument`: `from_mongo()` / `to_mongo()` — handles UUID ↔ MongoDB `_id` conversion
- `VectorBaseDocument`: `from_record()` / `to_point()` — converts between Pydantic models and Qdrant `PointStruct` / `Record` objects

*Expected answer:*
- `to_point()` extracts `id` and `embedding` from the payload dict, converts numpy arrays to lists, and constructs a `PointStruct(id, vector, payload)` — Qdrant needs vectors separate from payload
- `from_record()` reconstitutes the UUID from the record ID, merges payload attributes, and conditionally sets `embedding` only if the class defines that field (via `_has_class_attribute`)
- Both handle UUID serialization but for different backends with different expectations

---

**Q13. The `VectorBaseDocument.bulk_insert()` method auto-creates collections on first insert. Walk through this retry-on-failure pattern.**

*Actual code (lines 80-97 of `vector.py`):*
```python
@classmethod
def bulk_insert(cls, documents):
    try:
        cls._bulk_insert(documents)
    except exceptions.UnexpectedResponse:
        logger.info(f"Collection '{cls.get_collection_name()}' does not exist. Trying to create...")
        cls.create_collection()
        try:
            cls._bulk_insert(documents)
        except exceptions.UnexpectedResponse:
            logger.error(f"Failed to insert documents in '{cls.get_collection_name()}'.")
            return False
    return True
```

*Expected answer:*
- Try insert first (optimistic approach — collection likely exists)
- On `UnexpectedResponse` (collection doesn't exist), create it via `create_collection()` then retry
- If second attempt also fails, log error and return `False` — graceful degradation without crashing the pipeline
- `create_collection()` uses `EmbeddingModelSingleton().embedding_size` for vector dimension and `Distance.COSINE` as the metric, but only if `use_vector_index=True`

---

**Q14. Cleaned documents have `use_vector_index = False` while embedded chunks have `use_vector_index = True`. What does this flag control?**

*Actual code (`vector.py` line 188-194):*
```python
@classmethod
def _create_collection(cls, collection_name, use_vector_index=True):
    if use_vector_index is True:
        vectors_config = VectorParams(size=EmbeddingModelSingleton().embedding_size, distance=Distance.COSINE)
    else:
        vectors_config = {}
    return connection.create_collection(collection_name=collection_name, vectors_config=vectors_config)
```

*Expected answer:*
- `use_vector_index=True` creates a collection with a 384-dim COSINE vector index (for ANN search)
- `use_vector_index=False` creates a collection with no vector configuration — just payload storage
- Cleaned documents don't have embeddings, so they're stored as plain documents in Qdrant; embedded chunks need the vector index for similarity search

---

**Q15. `VectorBaseDocument` has `group_by_class()` and `group_by_category()` methods. Where and why are they used?**

*Actual code:*
```python
@classmethod
def group_by_class(cls, documents):
    return cls._group_by(documents, selector=lambda doc: doc.__class__)

@classmethod
def group_by_category(cls, documents):
    return cls._group_by(documents, selector=lambda doc: doc.get_category())
```

*Used in `load_to_vector_db.py`:*
```python
grouped_documents = VectorBaseDocument.group_by_class(documents)
for document_class, class_documents in grouped_documents.items():
    for documents_batch in utils.misc.batch(class_documents, size=4):
        document_class.bulk_insert(documents_batch)
```

*Expected answer:*
- `group_by_class` groups documents by their concrete Python class (e.g., `EmbeddedArticleChunk`, `EmbeddedPostChunk`)
- This is critical for `load_to_vector_db` because each class maps to a different Qdrant collection — you must insert `EmbeddedArticleChunk` documents into the `embedded_articles` collection, not into `embedded_posts`
- `group_by_category` groups by `DataCategory` enum — useful for analytics/metadata

---

## 5. Embeddings & NLP

**Q16. Study `EmbeddingModelSingleton` in `embeddings.py`. Why is `embedding_size` a `@cached_property` while `max_input_length` is a regular `@property`?**

*Actual code:*
```python
@cached_property
def embedding_size(self) -> int:
    dummy_embedding = self._model.encode("")
    return dummy_embedding.shape[0]

@property
def max_input_length(self) -> int:
    return self._model.max_seq_length
```

*Expected answer:*
- `embedding_size` requires encoding a dummy string and inspecting the output shape — an expensive operation. `@cached_property` ensures this is computed exactly once and then memoized
- `max_input_length` is a simple attribute lookup on the model object — nearly free, so caching is unnecessary
- This also means `embedding_size` is lazy — it's not computed until first accessed, avoiding unnecessary model inference during initialization

---

**Q17. The `CrossEncoderModelSingleton` exists alongside `EmbeddingModelSingleton`. What is the difference between a bi-encoder and a cross-encoder, and when would you use each in a RAG system?**

*Actual code:*
```python
class EmbeddingModelSingleton(metaclass=SingletonMeta):  # bi-encoder
    # encodes text independently → dense vector

class CrossEncoderModelSingleton(metaclass=SingletonMeta):  # cross-encoder
    # scores (query, passage) pairs jointly
    def __call__(self, pairs: list[tuple[str, str]], ...) -> ...:
        scores = self._model.predict(pairs)
```

*Expected answer:*
- **Bi-encoder** (EmbeddingModel): encodes query and documents *independently* → fast, scalable (encode once, search many times), used for initial retrieval
- **Cross-encoder** (CrossEncoder): takes a *(query, document)* pair as joint input → more accurate but O(n) per query, used for **reranking** a small candidate set
- RAG workflow: bi-encoder retrieves top-K candidates → cross-encoder reranks them for precision

---

**Q18. Why was `sentence-transformers/all-MiniLM-L6-v2` chosen? What are the tradeoffs of 384-dim vs. 768-dim embeddings?**

*Expected answer:*
- MiniLM-L6-v2: 22M params, fast inference, 384-dim output, 256 max tokens
- Tradeoffs: ~50% less storage + faster search vs. slightly lower semantic resolution
- For a RAG retrieval system with chunked text, the quality difference is usually negligible — retrieval accuracy is dominated by chunk quality, not embedding dimension

---

**Q19. The project uses different chunk sizes for different content types: Posts (250 tokens/25 overlap), Articles (sentence-aware 1000-2000 chars), Repositories (1500 tokens/100 overlap). Justify these choices.**

*Expected answer:*
- **Posts** are short social media content — smaller chunks preserve self-contained meaning, overlap ensures boundary content isn't lost
- **Articles** have narrative structure — sentence-aware splitting preserves argumentative coherence without breaking mid-sentence
- **Repositories** contain code functions/classes that form larger logical units — bigger chunks keep enough context for code understanding
- The overlap ratio (~10%) prevents information loss at chunk boundaries

---

**Q20. Explain the two-stage chunking strategy: `RecursiveCharacterTextSplitter` → `SentenceTransformersTokenTextSplitter`.**

*Expected answer:*
- **Stage 1** (`RecursiveCharacterTextSplitter`): splits long text on semantic boundaries (`\n\n`, then `\n`, then spaces) — produces paragraph-level chunks
- **Stage 2** (`SentenceTransformersTokenTextSplitter`): ensures each chunk fits within the embedding model's 256-token limit
- Why two stages? Stage 1 preserves semantic structure; Stage 2 enforces the hard technical constraint. Without Stage 1, you'd get arbitrary 256-token windows that split mid-sentence

---

## 6. Vector Database & Similarity Search

**Q21. What is cosine similarity? Why is it the default distance metric in the Qdrant collections? When might you prefer dot-product?**

*Actual code in `vector.py`:*
```python
vectors_config = VectorParams(size=EmbeddingModelSingleton().embedding_size, distance=Distance.COSINE)
```

*Expected answer:*
- `cos(θ) = (A·B) / (||A|| × ||B||)` — measures angle between vectors, invariant to magnitude
- Good for text retrieval because document length shouldn't affect relevance scoring
- Dot-product is preferred when magnitude carries meaning (e.g., document importance) or when vectors are already normalized (then cosine = dot product)

---

**Q22. Explain deterministic chunk IDs using `UUID(MD5(content))`. Why is this critical for pipeline idempotency?**

*Expected answer:*
- Hashing content → UUID means identical content always produces the same ID
- In Qdrant, `bulk_insert` uses `upsert` semantics — same ID = overwrite, not duplicate
- Result: running the pipeline multiple times produces identical state — no snowballing duplicates
- This is essential for production pipelines that may be retried on failure

---

**Q23. What is HNSW (Hierarchical Navigable Small World)? How does Qdrant use it for approximate nearest neighbor search?**

*Expected answer:*
- HNSW builds a multi-layered navigable graph: upper layers have long-range connections for coarse navigation, lower layers have short-range connections for fine-grained search
- Search: start at the top layer, greedily navigate to the nearest node, then descend to lower layers for refinement
- Tradeoffs: O(log n) search time, high recall (>95%), but requires memory for the graph structure
- Qdrant's `VectorParams` configures the index at collection creation time

---

## 7. Design Patterns in the Codebase

**Q24. Three design patterns — Strategy, Factory, Dispatcher — are used together in the preprocessing pipeline. Trace how they interact for the cleaning stage.**

*Actual code:*
1. **Strategy**: `CleaningDataHandler` (abstract) defines `clean()` interface → `PostCleaningHandler`, `ArticleCleaningHandler`, `RepositoryCleaningHandler` implement it
2. **Factory**: `CleaningHandlerFactory.create_handler(DataCategory)` returns the correct concrete handler
3. **Dispatcher**: `CleaningDispatcher.dispatch(document)` extracts the category, calls the factory, and invokes `handler.clean()`

*Expected answer:*
- Adding a new data type (e.g., LinkedIn) only requires: (1) new handler implementing `clean()`, (2) new elif in factory, (3) no changes to dispatcher
- This is the Open/Closed Principle — open for extension, closed for modification

---

**Q25. The codebase uses two different singleton implementations. Compare `__new__`-based singletons (Mongo/Qdrant connectors) vs. `SingletonMeta` metaclass (embedding models).**

*Actual code:*
- `infrastructure/db/mongo.py`: uses `__new__` override — stores instance on the class itself
- `networks/base.py`: `SingletonMeta` metaclass with thread-safe locking

*Expected answer:*
- `__new__` is simpler — overrides instance creation in the class itself, good for one-off cases
- `SingletonMeta` is a reusable metaclass applicable to *any* class — uses `threading.Lock()` for thread safety
- Thread safety matters for `EmbeddingModelSingleton` because the `ThreadPoolExecutor` in `query_data_warehouse` could trigger concurrent instantiation; database connectors are typically created before threading begins

---

**Q26. The `VectorBaseDocument` has a `collection_name_to_class()` recursive method. What problem does it solve?**

*Actual code (lines 244-258 of `vector.py`):*
```python
@classmethod
def collection_name_to_class(cls, collection_name: str) -> type["VectorBaseDocument"]:
    for subclass in cls.__subclasses__():
        try:
            if subclass.get_collection_name() == collection_name:
                return subclass
        except ImproperlyConfigured:
            pass
        try:
            return subclass.collection_name_to_class(collection_name)
        except ValueError:
            continue
    raise ValueError(f"No subclass found for collection name: {collection_name}")
```

*Expected answer:*
- Given a collection name string (e.g., `"embedded_articles"`), it traverses the entire subclass hierarchy to find the matching class (`EmbeddedArticleChunk`)
- This is a **reverse lookup** — useful when deserializing Qdrant records where you only know the collection name but need to instantiate the correct Python class
- The recursion handles multi-level inheritance (e.g., `VectorBaseDocument` → `EmbeddedChunk` → `EmbeddedArticleChunk`)

---

## 8. MLOps & Pipeline Orchestration

**Q27. Why use ZenML instead of Airflow or a custom script? What specific features does the codebase use?**

*Actual features used:*
- `@pipeline` and `@step` decorators for declarative pipeline definitions
- `get_step_context().add_output_metadata()` for per-step observability logging
- `Annotated[list, "raw_documents"]` for typed, named step outputs
- `--no-cache` flag for controlling step re-execution
- Pipeline caching for expensive steps (crawling, embedding)

*Expected answer:*
- ZenML separates pipeline logic from infrastructure — same code runs locally or on Kubernetes
- ML-native features: artifact versioning, experiment tracking integration, step caching, metadata logging
- The `add_output_metadata()` calls in every step provide audit trails for data validation (document counts, author distributions)

---

**Q28. Every pipeline step logs metadata via `get_step_context().add_output_metadata()`. What metadata is tracked and why?**

*Actual metadata tracked:*
- `query_data_warehouse`: document counts per collection, author names per category
- `clean_documents`: document counts per category, author names
- `chunk_and_embed`: chunking metadata (chunk size, overlap), embedding metadata, total chunk/embedded counts

*Expected answer:*
- This creates an audit trail for every pipeline run — critical for debugging data quality issues
- If retrieval quality drops, you can inspect metadata to see if document counts changed, if certain authors are missing, or if chunk sizes shifted
- ZenML stores this metadata per run, enabling historical comparison across runs

---

## 9. Model Training & Fine-Tuning (Planned)

**Q29. Explain QLoRA (Quantized Low-Rank Adaptation). Why is it the planned approach for fine-tuning?**

*Expected answer:*
- QLoRA = 4-bit quantized frozen base model + low-rank adapter matrices (LoRA, rank r ~8-64) trained in full precision
- Reduces memory by ~4x for the base model; only <1% of total parameters are trainable
- Enables fine-tuning 7B+ models on a single GPU — critical for a solo-builder project
- Math: weight update `ΔW = BA` where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r << min(d,k)

---

**Q30. The project plans SFT followed by DPO. Why this order? What is the Bradley-Terry model underlying DPO?**

*Expected answer:*
- SFT teaches the model to generate desired outputs (instruction → response mapping)
- DPO then aligns it with stylistic preferences (chosen vs. rejected pairs) using the Bradley-Terry model: `P(A ≻ B) = σ(s_A - s_B)`
- DPO directly optimizes the language model's log-probabilities as the scoring function, eliminating the need for a separate reward model (unlike RLHF)
- Order matters: DPO requires a capable baseline — it refines style, not capability

---

## 10. Inference & Deployment (Planned)

**Q31. Explain KV-cache in transformer inference. Why does it matter for autoregressive generation?**

*Expected answer:*
- Without KV-cache: at each generation step, recompute K and V matrices for all previous positions → O(n²)
- With KV-cache: store previously computed K/V pairs, each new token only computes its own K/V → per-token cost drops to O(n)
- This is the main optimization that makes real-time LLM serving feasible

---

**Q32. What is model quantization (INT8/INT4)? What quality-latency tradeoffs does it introduce?**

*Expected answer:*
- Reduces weight precision from FP16 → INT8 (2x smaller) or INT4 (4x smaller)
- Enables faster inference and lower memory usage
- Quality degradation: typically <1% for INT8, 2-5% for INT4 on complex reasoning
- Calibration techniques (e.g., GPTQ, AWQ) minimize quality loss by choosing optimal quantization parameters

---

## 11. Mathematical Foundations

**Q33. Derive the cosine similarity formula and explain why normalizing vectors to unit length makes cosine similarity equivalent to dot product.**

*Expected answer:*
- `cos(θ) = (A·B) / (||A|| × ||B||)`
- When ||A|| = ||B|| = 1 (unit vectors), denominator = 1, so `cos(θ) = A·B`
- Many vector DBs normalize embeddings at index time, converting expensive cosine operations into cheaper dot products

---

**Q34. What is KL divergence? How is it used in both DPO training and data drift detection?**

*Expected answer:*
- `D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))` — measures how distribution P diverges from reference Q
- **DPO**: constrains the aligned model to stay close to the SFT base (prevents catastrophic forgetting)
- **Drift detection**: compares current embedding/data distribution to training distribution — large KL triggers retraining

---

**Q35. Write the attention mechanism equation. What does `√d_k` scaling prevent?**

*Expected answer:*
- `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`
- Q = queries, K = keys, V = values
- Without `√d_k` scaling, dot products grow proportionally to `d_k` in high dimensions, pushing softmax into saturation (near-zero gradients) → training becomes unstable

---

## 12. Scenario-Based / Applied Judgment

**Q36. If retrieval quality drops after adding 10x more documents, what diagnostic and mitigation steps would you take?**

*Expected answer:*
1. Check chunk quality — run cleaning/chunking analysis on new docs
2. Evaluate embedding distribution shift — new docs may be out-of-domain for MiniLM
3. Tune retrieval: increase `k`, add reranking via `CrossEncoderModelSingleton`, try hybrid search (BM25 + dense)
4. Analyze HNSW index parameters (ef_construction, m) — may need tuning for larger collections
5. Consider domain-specific embedding model

---

**Q37. How would you add a new data source (e.g., LinkedIn posts) to the pipeline? List all code changes needed.**

*Expected answer (referencing actual files):*
1. `crawlers/linkedin.py` — new crawler extending `BaseCrawler`
2. `crawlers/dispatcher.py` — `register("https://linkedin.com", LinkedInCrawler)`
3. `domain/documents.py` — `LinkedInPostDocument(Document)` with LinkedIn-specific fields
4. `domain/cleaned_documents.py` — `CleanedLinkedInDocument(CleanedDocument)`
5. `domain/chunks.py` — `LinkedInChunk(Chunk)`
6. `domain/embedded_chunks.py` — `EmbeddedLinkedInChunk(EmbeddedChunk)`
7. `preprocessing/cleaning_data_handlers.py` — `LinkedInCleaningHandler`
8. `preprocessing/chunking_data_handlers.py` — `LinkedInChunkingHandler`
9. `preprocessing/embedding_data_handlers.py` — `LinkedInEmbeddingHandler`
10. `preprocessing/dispatchers.py` — add `DataCategory.LINKEDIN` to all 3 factories
11. `domain/types.py` — add `LINKEDIN = "linkedin"` to `DataCategory` enum

---

**Q38. The `EmbeddedChunk` class has a `to_context()` classmethod. What is its purpose in a RAG system?**

*Actual code (`embedded_chunks.py`):*
```python
@classmethod
def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"""
        Chunk {i + 1}:
        Type: {chunk.__class__.__name__}
        Platform: {chunk.platform}
        Author: {chunk.author_full_name}
        Content: {chunk.content}\n
        """
    return context
```

*Expected answer:*
- This formats retrieved chunks into a structured text string that gets injected into the LLM prompt as context
- Includes metadata (type, platform, author) alongside content — this gives the LLM attribution information and source awareness
- This is the bridge between the retrieval step and the generation step in RAG

---

**Q39. The `DataCategory` enum includes future categories like `PROMPT`, `QUERIES`, `INSTRUCT_DATASET_SAMPLES`, `PREFERENCE_DATASET_SAMPLES`. What pipeline stages will these support?**

*Actual code (`types.py`):*
```python
class DataCategory(StrEnum):
    PROMPT = "prompt"
    QUERIES = "queries"
    INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"
    INSTRUCT_DATASET = "instruct_dataset"
    PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"
    PREFERENCE_DATASET = "preference_dataset"
    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"
```

*Expected answer:*
- `PROMPT` / `QUERIES`: RAG inference pipeline — user queries stored for retrieval and monitoring
- `INSTRUCT_DATASET` / `INSTRUCT_DATASET_SAMPLES`: SFT training pipeline — instruction-response pairs
- `PREFERENCE_DATASET` / `PREFERENCE_DATASET_SAMPLES`: DPO training pipeline — chosen/rejected pairs
- The enum is designed upfront to support the full FTI lifecycle without modification

---

**Q40. If you had to make this system production-ready for 1000 concurrent users, what architectural changes would you prioritize?**

*Expected answer:*
1. Async FastAPI with connection pooling for MongoDB and Qdrant
2. Qdrant cluster mode with sharding
3. Redis caching layer for frequent/similar queries
4. Load balancer in front of inference endpoints
5. Horizontal scaling of the embedding service (separate from the LLM)
6. Rate limiting and request queuing
7. Prometheus/Grafana monitoring with alerting on latency and error rates

---

> 💡 **Tip for the Interviewer:**
> - Questions **1-7** test **data engineering & ETL** understanding
> - Questions **8-15** test **feature pipeline & domain modeling** depth
> - Questions **16-23** test **NLP & vector search** knowledge
> - Questions **24-28** test **software engineering & MLOps** maturity
> - Questions **29-35** test **ML theory & math foundations**
> - Questions **36-40** test **applied judgment & system thinking**
