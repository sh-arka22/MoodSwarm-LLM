# Development Guide

## Prerequisites

**Accounts needed:** HuggingFace, AWS (SageMaker), Comet ML, MongoDB Atlas (optional), Qdrant Cloud (optional)

**Local tools:** Docker, Python 3.11 (via pyenv), Poetry 1.8+, Node.js 18+ (for frontend), ZenML 0.74.0

## Setup

```bash
pyenv local 3.11.8
poetry install
docker compose up -d
poetry run zenml init
```

## Full Command Reference

### Infrastructure

```bash
docker compose up -d                  # Start MongoDB + Qdrant
docker compose down                   # Stop services
docker ps                             # Verify containers
poetry run zenml status               # ZenML stack health
poetry run zenml pipeline list        # List pipelines
```

### Data Pipelines

```bash
# ETL
poetry run python -m tools.run --run-smoke-test --no-cache
poetry run python -m tools.run --run-etl --no-cache
poetry run python -m tools.data_warehouse --export-raw-data
poetry run python -m tools.data_warehouse --import-raw-data

# Feature Engineering
poetry run python -m tools.run --run-feature-engineering --no-cache
```

### Qdrant Inspection

```bash
poetry run python -m tools.qdrant_inspect list-collections
poetry run python -m tools.qdrant_inspect collection-stats embedded_articles
poetry run python -m tools.qdrant_inspect sample embedded_articles
poetry run python -m tools.qdrant_inspect search embedded_articles --query "RAG systems"
```

### Analysis & Search

```bash
poetry run python -m tools.chunk_analysis
poetry run python -m tools.search_test -q "your query here" --k 3
```

### RAG Retrieval

```bash
poetry run python -m tools.rag -q "How do RAG systems work?" --k 3
poetry run python -m tools.rag -q "My name is Arkajyoti Saha. Write about LLMs." --k 9
poetry run python -m tools.rag -q "What are best practices?" --mock
poetry run python -m tools.rag_tuning
poetry run python -m tools.rag_eval --k 3
poetry run python -m tools.rag_eval --k 6
```

### Dataset & Training

```bash
# Dataset inspection
poetry run python -m tools.dataset_inspect stats
poetry run python -m tools.dataset_inspect stats --type preference
poetry run python -m tools.dataset_inspect evaluate --n 10
poetry run python -m tools.dataset_inspect quality --deep

# Push to HuggingFace
poetry run python -m tools.push_dataset --dry-run
poetry run python -m tools.push_dataset
poetry run python -m tools.push_dataset --dataset-type preference

# Training
poetry run python -m tools.sft_report
poetry run python -m tools.run --run-training --no-cache
poetry run python -m tools.run --run-evaluation --no-cache
poetry run python -m tools.model_compare --n 20
```

### Inference & Deployment

```bash
poetry run python -m tools.deploy_endpoint create      # ~$1.20/hr
poetry run python -m tools.deploy_endpoint status
poetry run python -m tools.deploy_endpoint delete       # Stops billing
poetry run python -m tools.ml_service                   # FastAPI on :8000

# Test endpoint
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "How do RAG systems work?"}'
```

### CI/CD & Testing

```bash
poetry poe lint-check
poetry poe format-check
poetry poe test
poetry run pre-commit run --all-files
docker build -t moodswarm .
```

### Frontend

```bash
cd frontend
npm install
npx expo start          # Press 'w' for web, 'i' for iOS
```

## File Tree

```
llm_engineering/
├── settings.py                              # Pydantic BaseSettings, loads from .env
├── domain/
│   ├── types.py                             # DataCategory enum
│   ├── exceptions.py                        # LLMTwinException, ImproperlyConfigured
│   ├── documents.py                         # UserDocument, ArticleDocument, PostDocument, RepositoryDocument
│   ├── conversations.py                     # ConversationDocument, MessageDocument
│   ├── cleaned_documents.py                 # CleanedPostDocument, CleanedArticleDocument, CleanedRepositoryDocument
│   ├── chunks.py                            # PostChunk, ArticleChunk, RepositoryChunk
│   ├── embedded_chunks.py                   # EmbeddedPostChunk, EmbeddedArticleChunk, EmbeddedRepositoryChunk
│   ├── queries.py                           # Query, EmbeddedQuery
│   ├── inference.py                         # ABC: Inference, DeploymentStrategy
│   └── base/
│       ├── nosql.py                         # NoSQLBaseDocument ODM (MongoDB)
│       └── vector.py                        # VectorBaseDocument ODM (Qdrant)
├── application/
│   ├── crawlers/                            # URL dispatcher + GitHub/Medium/Custom crawlers
│   ├── networks/                            # Singleton embedding + cross-encoder models
│   ├── preprocessing/                       # Clean/chunk/embed handlers + dispatchers
│   ├── rag/                                 # SelfQuery, QueryExpansion, Reranking, ContextRetriever
│   └── utils/                               # batch(), flatten(), compute_num_tokens()
├── infrastructure/
│   ├── inference_pipeline_api.py            # FastAPI endpoints (RAG + conversations)
│   ├── opik_utils.py                        # Opik/Comet monitoring
│   ├── db/                                  # MongoDB + Qdrant connectors
│   └── aws/deploy/                          # SageMaker deploy strategy + config
└── model/
    ├── finetuning/                           # SFT/DPO entry point (Unsloth QLoRA)
    ├── inference/                            # SageMaker client + InferenceExecutor
    └── evaluation/                           # LLM-as-judge scoring

pipelines/                                    # ZenML pipeline definitions
├── smoke_test.py                            # Health check
├── digital_data_etl.py                      # ETL
├── feature_engineering.py                   # Clean → Chunk → Embed → Qdrant
├── generate_datasets.py                     # Instruct + preference datasets
├── training.py                              # SageMaker SFT/DPO
└── evaluating.py                            # Model evaluation

steps/                                        # Reusable ZenML pipeline steps
├── etl/                                     # get_or_create_user, crawl_links
├── feature_engineering/                     # query, clean, chunk+embed, load to Qdrant
├── generate_datasets/                       # query feature store, create prompts, generate, push
├── training/                                # SageMaker training step
└── evaluating/                              # SageMaker evaluation step

tools/                                        # CLI utilities
├── run.py                                   # Pipeline runner (--run-etl, --run-training, etc.)
├── data_warehouse.py                        # MongoDB ↔ JSON export/import
├── qdrant_inspect.py                        # Qdrant collection inspection
├── rag.py                                   # RAG retrieval CLI
├── rag_tuning.py / rag_eval.py              # RAG parameter tuning + evaluation
├── dataset_inspect.py / push_dataset.py     # Dataset tools
├── deploy_endpoint.py                       # SageMaker lifecycle CLI
└── ml_service.py                            # FastAPI uvicorn launcher

frontend/                                     # React Native (Expo) chat UI
├── app/                                     # expo-router pages
│   ├── _layout.tsx                          # Drawer navigation
│   ├── index.tsx                            # Redirect to /chat/new
│   └── chat/[id].tsx                        # Chat screen
├── components/                              # UI components
│   ├── ChatBubble.tsx                       # Message bubble
│   ├── ChatInput.tsx                        # Text input + send
│   ├── ConversationList.tsx                 # Thread sidebar
│   ├── ConversationItem.tsx                 # Single thread row
│   ├── RenameModal.tsx                      # Rename dialog
│   └── EmptyState.tsx                       # Empty chat placeholder
├── constants/theme.ts                       # Dark theme colors
└── lib/
    ├── types.ts                             # TypeScript interfaces
    └── api.ts                               # Backend API client

configs/                                      # Pipeline YAML configs
tests/                                        # Unit + integration tests
docs/                                         # Documentation
.github/workflows/                            # CI (lint+test) + CD (Docker+ECR)
```

## Week-by-Week Build Log

### Week 1 — Foundation & Environment Setup
- Cloned reference repo, set up Python/Poetry/Docker/ZenML
- Configured env vars, ran smoke test pipeline
- **Milestone:** Local dev environment fully operational

### Week 2 — Data Collection & ETL Pipeline
- Built crawler dispatcher (GitHub, Medium, Custom Article)
- 3/3 articles crawled, deduplication verified, retry logic added
- Data warehouse round-trip (MongoDB ↔ JSON) verified
- **Milestone:** ETL runs end-to-end with predictable artifacts

### Week 3 — RAG Feature Pipeline & Vector Store
- Built Qdrant ODM, embedding singleton, cleaning/chunking/embedding dispatchers
- Feature engineering pipeline: 3 articles → 3 cleaned + 26 embedded chunks
- Deterministic chunk IDs, idempotent upserts, semantic search working
- **Milestone:** All documents cleaned, chunked, embedded in Qdrant

### Week 4 — RAG Retrieval & Baseline Quality
- Implemented SelfQuery, QueryExpansion, CrossEncoder reranking
- ContextRetriever orchestrator with parallel search
- Baseline: Recall@3=0.43, Recall@6=0.60, MRR@3=1.0
- **Milestone:** Consistent retrieval with measured baselines

### Week 5 — Instruction Dataset & SFT Training
- Generated 65-sample instruction dataset, pushed to HuggingFace
- SFT training: Unsloth QLoRA, 3 epochs, loss 1.12→0.49, ~$4.50
- Model: `saha2026/TwinLlama-3.1-8B`
- **Milestone:** SFT baseline trained and checkpointed

### Week 6 — DPO Preference Alignment & Evaluation
- Generated 80-sample preference dataset
- DPO training on SFT model, ~$0.60
- LLM-as-judge evaluation, SFT vs DPO comparison tool
- Model: `saha2026/TwinLlama-3.1-8B-DPO`
- **Milestone:** DPO-improved model with comparative evaluation

### Week 7 — Inference Optimization & Deployment
- SageMaker endpoint with TGI v2.4.0, INT8 quantization
- FastAPI `/rag` endpoint with Opik monitoring
- Deploy/delete/status CLI for endpoint lifecycle
- **Milestone:** End-to-end RAG inference via FastAPI + SageMaker

### Week 8 — MLOps, CI/CD & Capstone
- Pre-commit hooks (ruff + gitleaks), pytest suite, Poe tasks
- GitHub Actions CI/CD (lint → test → Docker → ECR)
- Operations runbook, Docker containerization
- React Native chat UI with conversation management
- **Milestone:** Production-ready with CI/CD, testing, and chat interface
