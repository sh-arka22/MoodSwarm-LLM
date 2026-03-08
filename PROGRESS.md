# MoodSwarm — Weekly Progress Report

**Project:** Production-ready LLM system following *The LLM Engineer's Handbook*
**Builder:** Arkajyoti Saha (solo)
**Timeline:** 8 weeks | **Current:** Week 5 (Day 4 of 7)
**Reference:** Paul Iusztin & Maxime Labonne, *The LLM Engineer's Handbook* (11 chapters)

---

## Architecture Overview

```
Data Collection → ETL Pipeline → Feature Store → Model Training → Inference & Evaluation
      ↓               ↓               ↓              ↓                    ↓
  Crawlers         MongoDB         Qdrant       HuggingFace          FastAPI /rag
  (GitHub,        (raw docs)      (vectors)      (SFT/DPO)           AWS SageMaker
   Medium,
   Custom)
```

**Stack:** Python 3.11 | Poetry | Docker (MongoDB + Qdrant) | ZenML | OpenAI | HuggingFace | LangChain

---

## Week 1 — Foundation & Environment Setup (DONE)

**Chapters:** 1 (LLM Twin Architecture), 2 (Tooling & Installation)
**Phase:** 1A — Infrastructure

### What We Built
- Python 3.11.8 via pyenv with Poetry dependency management
- Docker Compose orchestrating MongoDB (document store) and Qdrant (vector DB)
- ZenML ML pipeline framework initialized with default local stack
- Pydantic `BaseSettings` configuration loading from `.env`
- Smoke test pipeline running end-to-end through ZenML

### Key Decisions
- **Domain-Driven Design (DDD)** package layout: `domain/`, `application/`, `infrastructure/`, `model/`
- MongoDB for raw unstructured documents, Qdrant for vector embeddings
- ZenML for pipeline orchestration with `@pipeline` and `@step` decorators

### Artifacts
| Artifact | Status |
|----------|--------|
| Docker services (MongoDB + Qdrant) | Healthy |
| ZenML stack registered | Active |
| Smoke test pipeline | Passing |
| Environment config (`.env` + `settings.py`) | Verified |

---

## Week 2 — Data Collection & ETL Pipeline (DONE)

**Chapters:** 3 (Data Engineering)
**Phase:** 1B — Data Ingestion

### What We Built
- **Crawler framework** with 3 specialized crawlers:
  - `GitHubCrawler` — repository content extraction
  - `MediumCrawler` — blog article scraping
  - `CustomArticleCrawler` — generic article fallback
- **Dispatcher pattern** — URL regex routing to the correct crawler class
- **MongoDB ODM** — `NoSQLBaseDocument` with `from_mongo()`/`to_mongo()` handling UUID-to-`_id` conversion
- **ETL pipeline** (`digital_data_etl`) — user lookup, crawl links, store in MongoDB
- **Deduplication** — each crawler checks `find(link=link)` before scraping; reruns skip existing docs
- **Retry logic** — `tenacity` decorator with 3 attempts, exponential backoff on `ConnectionError`/`TimeoutError`
- **Data warehouse tool** — `tools/data_warehouse.py` for MongoDB-to-JSON round-trip export/import

### Results
- 3 articles crawled for author Paul Iusztin (14K + 8K + 7K characters)
- Deduplication verified — rerun skipped all 3 existing docs (zero duplicates)
- Data warehouse round-trip: export to JSON, import back, matching document counts
- ZenML metadata logging per-domain success/failure counts

### Key Patterns Established
- Builder pattern: `CrawlerDispatcher.register()` maps regex to crawler class
- Singleton MongoDB connector via `__new__` — one connection per process
- Retry with backoff for all external HTTP calls

---

## Week 3 — RAG Feature Pipeline & Vector Store (DONE)

**Chapters:** 4 (RAG Feature Pipeline)
**Phase:** 2A — Feature Engineering

### What We Built (7 days)

**Day 1 — Infrastructure & Domain Models**
- Qdrant database connector (singleton, supports local Docker + Qdrant Cloud)
- `VectorBaseDocument` ODM (270 lines) — `to_point()`/`from_record()`, auto-creates collections on first insert
- 9 domain models: 3 cleaned (posts/articles/repos), 3 chunks, 3 embedded chunks
- `EmbeddingModelSingleton` wrapping `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 256 max tokens)

**Day 2 — Preprocessing Pipeline**
- Full clean → chunk → embed pipeline with Strategy + Dispatcher pattern
- Cleaning: join content dict → regex strip special chars → collapse whitespace
- Chunking per document type:
  - Posts: 250 tokens, 25 overlap
  - Articles: 1000-2000 chars sentence-aware
  - Repositories: 1500 tokens, 100 overlap
- 2-stage chunking: `RecursiveCharacterTextSplitter` → `SentenceTransformersTokenTextSplitter` (cap at 256 tokens)
- Deterministic chunk IDs via `UUID(MD5(content))` for idempotent upserts

**Day 3 — ZenML Pipeline**
- 4 pipeline steps: query data warehouse → clean → chunk+embed → load to Qdrant
- Feature engineering pipeline definition with YAML config
- CLI integration: `--run-feature-engineering`

**Day 4 — End-to-End Run**
- Pipeline: 3 articles → 3 cleaned docs + 26 embedded chunks (384-dim COSINE)
- Built `tools/qdrant_inspect.py` — list collections, stats, sample points, search
- Fixed bug: `connection.search()` doesn't exist in qdrant-client; switched to `query_points()`

**Day 5 — Idempotency & Validation**
- Re-run produces identical counts (idempotent via deterministic UUIDs)
- Built `tools/chunk_analysis.py` — char/token distribution analysis
- Fixed bug: article chunks exceeded 256 tokens (380-443); added token-capping 2nd stage
- After fix: 13 → 26 chunks, all within 256-token limit

**Day 6 — Query Model & Search**
- `Query` and `EmbeddedQuery` domain models — same embedding flow as chunks
- `QueryEmbeddingHandler` + `EmbeddingDispatcher` extended for QUERIES category
- Built `tools/search_test.py` — semantic search across all embedded collections

**Day 7 — Polish**
- `ruff check .` — all lint checks passed
- Full integration test: ETL → feature engineering → semantic search — all PASS

### Final State
| Collection | Points | Vectors | Distance |
|-----------|--------|---------|----------|
| `cleaned_articles` | 3 | None (payload only) | — |
| `embedded_articles` | 26 | 384-dim | COSINE |

---

## Week 4 — RAG Retrieval & Baseline Quality (DONE)

**Chapters:** 9 (RAG Inference Pipeline)
**Phase:** 2B — Retrieval System

### What We Built (7 days)

**Day 1 — Base Classes & Templates**
- `PromptTemplateFactory` (ABC) and `RAGStep` (ABC with mock flag)
- `QueryExpansionTemplate` and `SelfQueryTemplate` using LangChain `PromptTemplate`
- Added deps: `langchain-openai`, `opik`

**Day 2 — Pre-Retrieval: SelfQuery + QueryExpansion**
- **SelfQuery:** Extracts author name from natural language query via OpenAI `gpt-4o-mini` → `split_user_full_name()` → MongoDB user lookup → enriches `Query.author_id` for filtered vector search
- **QueryExpansion:** Generates N alternative query versions via OpenAI → splits by `#next-question#` separator → returns `list[Query]` preserving original ID
- Both use LangChain LCEL: `prompt | model` chain composition with `ChatOpenAI(temperature=0)`

**Day 3 — Post-Retrieval: Reranker + Orchestrator**
- **Reranker:** `CrossEncoderModelSingleton` (`ms-marco-MiniLM-L-4-v2`) scores `(query, chunk)` pairs → sorts by relevance score → returns top-k
- **ContextRetriever:** Full RAG orchestrator:
  1. SelfQuery → extract author, enrich query
  2. QueryExpansion → N diverse query variants
  3. Parallel search via `ThreadPoolExecutor` (per-variant, per-collection)
  4. Flatten → set dedup (by UUID) → Rerank → top-k results

**Day 4 — CLI & E2E Testing**
- Built `tools/rag.py` — full RAG CLI with `--query`, `--k`, `--expand-to-n`, `--mock`
- 4 test scenarios verified:
  - Mock mode (no API calls) — PASS
  - Real, no author mentioned → 2 chunks retrieved — PASS
  - Real, correct author (Paul) → 1 chunk with author filter — PASS
  - Real, wrong author → 0 chunks (expected) — PASS

**Day 5 — Performance Tuning**
- Built `tools/rag_tuning.py` — stage latency profiling + parameter sweep

| Stage | Latency |
|-------|---------|
| SelfQuery (OpenAI) | ~1700ms |
| QueryExpansion (OpenAI) | ~1460ms |
| Embedding (local MiniLM) | ~155ms |
| Vector search (Qdrant) | ~16ms |
| Reranking (CrossEncoder) | ~276ms |

- Optimal defaults: `k=3`, `expand_to_n=3` (~2s warm latency)
- OpenAI calls dominate at ~3200ms combined (93% of total)

**Day 6 — Baseline Metrics**
- Built `tools/rag_eval.py` with 7 curated test queries (2-4 expected chunks each)

| Metric | K=3 | K=6 |
|--------|-----|-----|
| Recall@K | 0.429 | 0.595 |
| MRR | 1.000 | 0.857 |

- MRR@3 = 1.0 means the reranker always puts a relevant result first
- Recall@3 = 0.43 is limited by small k per collection (`k//3 = 1` per type)

**Day 7 — Polish**
- `ruff check .` — all lint checks passed
- Full chain integration: ETL → feature engineering → RAG retrieval — all PASS

---

## Week 5 — Instruction Dataset & SFT Training (IN PROGRESS)

**Chapters:** 5 (Supervised Fine-Tuning)
**Phase:** 3A — Training Data + Fine-Tuning

### What We've Built (Days 1-4 of 7)

**Day 1 — Domain Models & Dependencies**
- Domain models: `DatasetType`, `InstructDatasetSample`, `PreferenceDatasetSample`, `InstructDataset`, `PreferenceDataset`, `TrainTestSplit`, `build_dataset()` factory
- `Prompt` and `GenerateDatasetSamplesPrompt` domain models
- Dataset generation module: `DatasetGenerator` (ABC) → `InstructionDatasetGenerator` / `PreferenceDatasetGenerator`
- LangChain LCEL: `llm | parser` chain with `.batch()` for parallel LLM calls (batches of 24)
- `ListPydanticOutputParser` for JSON array → list of Pydantic objects
- Filtering: `filter_short_answers()` (min 100 chars) + `filter_answer_format()` (uppercase start, `.!?` ending)
- Added deps: `tiktoken`, `datasets`, `scikit-learn`
- Fixed: Python 3.11.8 rebuilt with lzma/xz support for `datasets` library

**Day 2 — ZenML Pipeline + CLI**
- 5 pipeline steps: query feature store → create prompts → generate instruction dataset → generate preference dataset → push to HuggingFace
- Pipeline definition with configs for both instruction and preference datasets
- CLI integration: `--run-generate-instruct-datasets`, `--run-generate-preference-datasets`
- E2E mock verified: 3 articles → 13 prompts → 39 mock samples → 35/4 train/test split

**Day 3 — Real Dataset Generation**
- Generated instruction dataset via OpenAI `gpt-4o-mini`:
  - Input: 3 cleaned articles → 13 extracts (1000-2000 chars) → 13 prompts
  - Output: **65 instruction-answer pairs** (58 train + 7 test)
  - Time: 22.9s total, 1.8s per prompt
  - Cost: ~$0.004
- Built `tools/dataset_inspect.py` with 4 commands: `stats`, `samples`, `quality`, `generate`

| Metric | Value |
|--------|-------|
| Total samples | 65 |
| Train / Test | 58 / 7 (10.8%) |
| Unique instructions | 65/65 (100%) |
| Empty fields | 0 |
| Short answers (<50 chars) | 0 |
| Format issues | 0 |
| Keyword violations | 7 (minor — "course"/"system" mentions) |

**Day 4 — Data Quality Evaluation & Statistics**
- Built LLM-as-judge evaluation module (`llm_engineering/model/evaluation/evaluate.py`)
  - GPT-4o-mini scores each sample on accuracy (1-3) and style (1-3)
  - Parallel evaluation via `ThreadPoolExecutor` (4 threads, batches of 5)
- Added deep quality checks to `dataset_inspect.py`:
  - Near-duplicate detection (character n-gram Jaccard similarity)
  - Train/test contamination check
  - Vocabulary diversity analysis (type-token ratio, bigrams)

**Full Dataset Evaluation Results (65 samples):**

| Metric | Value |
|--------|-------|
| Accuracy avg | 2.23 / 3.0 |
| Style avg | 2.12 / 3.0 |
| Accuracy distribution | Poor=1, Good=48, Excellent=16 |
| Style distribution | Poor=0, Good=57, Excellent=8 |
| Near-duplicate pairs | 0 |
| Train/test contamination | 0 |
| Instruction vocab (TTR) | 219 words (0.33) |
| Answer vocab (TTR) | 1303 words (0.281) |
| Evaluation cost | ~$0.017 |

### Remaining (Days 5-7)
- Day 5: Push dataset to HuggingFace + SFT training setup (QLoRA config)
- Day 6: Fine-tune Llama 3.1 8B with QLoRA
- Day 7: Evaluate SFT model + Comet ML tracking + docs update

---

## Weeks 6-8 — Upcoming

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| 6 | DPO + Evaluation | Preference dataset, DPO training on SFT baseline, Ragas/ARES evaluation, SFT vs DPO comparison |
| 7 | Deployment | Model quantization (GGUF/GPTQ), AWS SageMaker endpoint, FastAPI `/rag` microservice |
| 8 | MLOps & Capstone | CI/CD with GitHub Actions, Docker containers, Opik monitoring, runbooks, final evaluation |

---

## Cumulative Stats

| Metric | Value |
|--------|-------|
| Python files | ~90 |
| Lines of code | ~4500 |
| ZenML pipelines | 4 (smoke test, ETL, feature engineering, dataset generation) |
| Pipeline steps | 15 |
| CLI tools | 9 (run, data_warehouse, qdrant_inspect, chunk_analysis, search_test, rag, rag_tuning, rag_eval, dataset_inspect) |
| MongoDB documents | 3 articles |
| Qdrant collections | 2 (cleaned_articles, embedded_articles) |
| Embedded chunks | 26 (384-dim COSINE) |
| Instruction dataset | 65 samples (58 train / 7 test) |
| Total OpenAI cost | ~$0.025 |

---

## Bugs Found & Fixed

| Week | Bug | Fix |
|------|-----|-----|
| W3D4 | `connection.search()` doesn't exist in qdrant-client | Changed to `connection.query_points()` |
| W3D5 | Article chunks exceeded 256 token limit (380-443 tokens) | Added `SentenceTransformersTokenTextSplitter` as 2nd stage |
| W4D1 | `setuptools` v82 removed `pkg_resources` breaking ZenML | Pin `setuptools<82` |
| W5D1 | Python 3.11.8 missing lzma module (`datasets` import fails) | Rebuilt Python with xz headers |

---

## Key Tools & Commands

```bash
# Environment
poetry install && poetry run pip install "setuptools<82"

# Docker services
docker compose up -d                    # Start MongoDB + Qdrant

# Pipelines
poetry run python -m tools.run --run-smoke-test --no-cache
poetry run python -m tools.run --run-etl --no-cache
poetry run python -m tools.run --run-feature-engineering --no-cache
poetry run python -m tools.run --run-generate-instruct-datasets --no-cache

# RAG retrieval
poetry run python -m tools.rag -q "How do RAG systems work?" --k 3
poetry run python -m tools.rag_eval --k 3
poetry run python -m tools.rag_tuning

# Dataset inspection
poetry run python -m tools.dataset_inspect stats
poetry run python -m tools.dataset_inspect quality --deep
poetry run python -m tools.dataset_inspect evaluate --n 65
poetry run python -m tools.dataset_inspect samples --n 5

# Lint
poetry run ruff check .
```
