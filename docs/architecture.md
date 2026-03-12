# Architecture Deep Dive

## System Overview

MoodSwarm follows the **Feature-Training-Inference (FTI)** pipeline architecture, a production pattern for LLM systems that cleanly separates data processing, model training, and serving.

```
Data Collection → ETL Pipeline → Feature Store → Model Training → Inference & Evaluation
      ↓               ↓               ↓              ↓                    ↓
  External          MongoDB         Qdrant       HuggingFace          FastAPI /rag
  sources        (raw docs)      (vectors)      (SFT/DPO)           AWS SageMaker
```

## Package Layout (Domain-Driven Design)

```
llm_engineering/
├── domain/           # Pure data models, no I/O
├── application/      # Business logic (crawlers, preprocessing, RAG)
├── infrastructure/   # External system adapters (DB, API, AWS)
└── model/            # ML-specific (training, inference, evaluation)
```

## Data Flow

### 1. ETL Pipeline (Weeks 1-2)

```
External Sources ──► CrawlerDispatcher ──► MongoDB
   GitHub                 │
   Medium           URL regex routing
   Substack         to specialized crawlers
```

**Key patterns:**
- **Crawler dispatch:** Builder pattern — `CrawlerDispatcher.register()` maps URL regex → crawler class; unmatched URLs fall back to `CustomArticleCrawler`
- **MongoDB ODM:** `NoSQLBaseDocument` with `from_mongo`/`to_mongo` handles UUID ↔ `_id` conversion
- **Deduplication:** Each crawler checks `self.model.find(link=link)` before scraping — reruns skip existing docs
- **Retry:** `tenacity` decorator on `extract` — 3 attempts, exponential backoff, retries on `ConnectionError`/`TimeoutError`/`OSError`
- **MongoDB singleton:** `MongoDatabaseConnector` uses `__new__` pattern — one connection per process
- **ZenML metadata:** `crawl_links` step logs per-domain success/failure counts as step metadata

### 2. Feature Engineering Pipeline (Week 3)

```
MongoDB docs ──► Clean ──► Chunk ──► Embed ──► Qdrant
                  │          │         │
           regex strip   2-stage   all-MiniLM-L6-v2
           whitespace    split     384-dim vectors
```

**Key patterns:**
- **Qdrant ODM:** `VectorBaseDocument` with `to_point`/`from_record`, `Config.name`/`Config.category`/`Config.use_vector_index` — auto-creates collections on first `bulk_insert`
- **Embedding singleton:** `EmbeddingModelSingleton` uses thread-safe `SingletonMeta` metaclass — wraps `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 256 max tokens)
- **Strategy + Dispatcher:** Factory creates per-type handler, Dispatcher routes by `DataCategory` — same pattern for all 3 stages (clean/chunk/embed) + queries
- **Chunking per type:** Posts = 250 tokens / 25 overlap, Articles = 1000-2000 chars sentence-aware, Repos = 1500 tokens / 100 overlap
- **2-stage chunking:** `RecursiveCharacterTextSplitter` (split on `\n\n`) → `SentenceTransformersTokenTextSplitter` (cap at 256 tokens)
- **Deterministic chunk IDs:** `UUID(MD5(content))` — enables idempotent Qdrant upserts on re-runs

### 3. RAG Retrieval Pipeline (Week 4)

```
User Query
    │
    ├──► SelfQuery (extract author → MongoDB lookup → filter)
    ├──► QueryExpansion (N diverse variants via OpenAI)
    ├──► Parallel Vector Search (Qdrant, per-category k/3)
    ├──► Flatten + Deduplicate
    └──► CrossEncoder Reranking → Top-K Context
```

**Key patterns:**
- **RAG base:** `PromptTemplateFactory` (ABC) + `RAGStep` (ABC with `mock` flag) — all RAG steps support mock mode for testing without API calls
- **SelfQuery:** Extracts author name via OpenAI `gpt-4o-mini` → `split_user_full_name()` → `UserDocument.get_or_create()` → enriches `Query.author_id` for filtered vector search
- **QueryExpansion:** Generates N alternative queries via OpenAI → splits by `#next-question#` separator → returns `list[Query]` preserving original ID
- **LangChain LCEL:** Both use `prompt | model` chain composition with `ChatOpenAI(temperature=0)`
- **Reranker:** `CrossEncoderModelSingleton` (`ms-marco-MiniLM-L-4-v2`) scores `(query, chunk)` pairs → sorts by score → returns top-k
- **ContextRetriever:** Full orchestrator — SelfQuery → QueryExpansion → parallel search (ThreadPoolExecutor) → flatten → set dedup → Rerank
- **Filtered search:** If `author_id` present, applies Qdrant `FieldCondition(key="author_id", match=MatchValue(...))` filter
- **Per-category k split:** `limit=k//3` per collection (posts/articles/repos) → merges results

**Baseline metrics:**
- Recall@3 = 0.43, Recall@6 = 0.60
- MRR@3 = 1.0, MRR@6 = 0.86
- Latency: OpenAI ~3.2s, Qdrant ~16ms, CrossEncoder ~276ms

### 4. Fine-Tuning Pipeline (Weeks 5-6)

```
Qdrant (cleaned docs) ──► Dataset Generation ──► SFT Training ──► DPO Training
                               │                     │                  │
                          LangChain LCEL         Unsloth QLoRA      DPOTrainer
                          GPT-4o-mini           Alpaca template     beta=0.5
```

**Key patterns:**
- **Dataset generation:** ABC `DatasetGenerator` → `InstructionDatasetGenerator` / `PreferenceDatasetGenerator` with LangChain LCEL `llm | parser` chains
- **SFT training:** Unsloth `FastLanguageModel` + QLoRA (rank=32, alpha=32, all attn+MLP projections), Alpaca template, `SFTTrainer`
- **DPO training:** `DPOTrainer(beta=0.5, ref_model=None)` — online DPO using base model as implicit reference
- **SageMaker flow:** `sagemaker_launcher.py` → `HuggingFace` estimator → `finetune.py` entry point on `ml.g5.2xlarge`
- **Model lineage:** `unsloth/Meta-Llama-3.1-8B` → SFT → `saha2026/TwinLlama-3.1-8B` → DPO → `saha2026/TwinLlama-3.1-8B-DPO`
- **LLM-as-judge:** `evaluate.py` scores (instruction, answer) pairs on accuracy (1-3) + style (1-3) via GPT-4o-mini

### 5. Inference & Deployment (Week 7)

```
FastAPI /rag
    │
    ├──► ContextRetriever.search() → Qdrant
    ├──► InferenceExecutor (Alpaca template)
    └──► SageMaker Endpoint (TGI, INT8, ml.g5.xlarge)
```

**Key patterns:**
- **Domain ABCs:** `Inference` (set_payload/inference) + `DeploymentStrategy` (deploy) — strategy pattern for swappable backends
- **SageMaker client:** `LLMInferenceSagemakerEndpoint` wraps boto3 `sagemaker-runtime` invoke_endpoint, JSON payload in/out
- **InferenceExecutor:** RAG prompt in **Alpaca template** (`### Instruction:` / `### Response:`) → LLM → extract `generated_text`. Fallback: retries with `return_full_text: True` if response is empty
- **Deploy infra:** `SagemakerHuggingfaceStrategy` → `DeploymentService` → `HuggingFaceModel.deploy()` with TGI v2.4.0 on ml.g5.xlarge
- **Config:** bitsandbytes INT8 quantization, 3072 input / 4096 total tokens, 150 max new tokens, temp=0.01
- **Opik monitoring:** `@opik.track` on call_llm_service + rag functions, logs model/token metadata

### 6. Chat UI & Conversations (Frontend)

```
React Native (Expo) ◄──► FastAPI Conversation API ◄──► MongoDB
    Drawer Nav                6 CRUD endpoints          conversations
    Chat Screen               CORS enabled              messages
    Thread Mgmt               Auto-titling
```

**Key patterns:**
- **Conversation persistence:** `ConversationDocument` + `MessageDocument` in MongoDB, following existing `NoSQLBaseDocument` ODM
- **Drawer navigation:** expo-router with `@react-navigation/drawer`, thread list in sidebar
- **Optimistic updates:** User messages appear immediately, replaced with server response
- **Auto-titling:** First message's text becomes the conversation title

## Environment Variables

Configured in `.env`, loaded via `llm_engineering/settings.py` (Pydantic BaseSettings):

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_HOST` | MongoDB connection string | `mongodb://llm_engineering:llm_engineering@127.0.0.1:27017` |
| `QDRANT_DATABASE_HOST` | Qdrant host | `localhost` |
| `QDRANT_DATABASE_PORT` | Qdrant port | `6333` |
| `OPENAI_API_KEY` | Query expansion, dataset generation | Required |
| `HUGGINGFACE_ACCESS_TOKEN` | Model/dataset hub access | Required |
| `COMET_API_KEY` | Experiment tracking (Opik) | Optional |
| `AWS_REGION` | SageMaker region | `eu-central-1` |
| `AWS_ACCESS_KEY` | AWS authentication | Required for deploy |
| `AWS_SECRET_KEY` | AWS authentication | Required for deploy |
| `HF_MODEL_ID` | Model to deploy | `saha2026/TwinLlama-3.1-8B-DPO` |
| `SAGEMAKER_ENDPOINT_INFERENCE` | Endpoint name | `twin` |

## Known Issues / Gotchas

- **setuptools pinning:** ZenML requires `pkg_resources` — keep `setuptools<82` (v82 removed `pkg_resources`)
- **USER_AGENT warning:** `langchain-community` emits a cosmetic warning about `USER_AGENT` env var — safe to ignore
- **Medium paywall:** MediumCrawler may fail on paywalled articles — use `CustomArticleCrawler` URLs or import backed-up data
- **torch pinning:** Must pin `torch>=2.0.0,<2.3.0` — newer versions lack macOS x86 wheels
- **Qdrant import triggers connection:** Importing `qdrant.py` connects immediately — Docker must be running
- **qdrant-client API:** No `connection.search()` — use `connection.query_points()` instead
- **Chunk logic changes = stale data:** Changing chunking params changes content hashes → different IDs → must delete collections and re-run
- **Opik noise:** `@opik.track` fires without COMET_API_KEY → noisy 401 warnings. Suppress with `OPIK_TRACK_DISABLE=true` env var
- **Alpaca template required:** Model produces zero tokens without `### Instruction:` / `### Response:` wrapping (trained with this format during SFT)
