# MoodSwarm

An end-to-end LLM Twin platform that learns a persona's writing style from their content (articles, repos, posts), then generates responses in that style via a RAG-powered chat interface.

Built with the **Feature-Training-Inference (FTI)** pipeline architecture following *The LLM Engineer's Handbook*.

## Architecture

```
React Native (Expo)           FastAPI                     AWS / Data
┌──────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│  Drawer Nav      │    │ POST /rag          │    │ MongoDB (documents) │
│  ├─ Thread List  │◄──►│ /conversations     │◄──►│ Qdrant  (vectors)   │
│  └─ Chat Screen  │    │ /conversations/:id │    │ SageMaker (LLM)     │
│    ├─ Messages   │    │   /messages        │    │ HuggingFace (model) │
│    └─ Input Bar  │    └────────────────────┘    └─────────────────────┘
└──────────────────┘       localhost:8000
   localhost:8081
```

**Pipeline:** Web Crawling → MongoDB → Clean/Chunk/Embed → Qdrant → SFT/DPO Training → SageMaker Endpoint → FastAPI + Chat UI

## Quick Start

### Prerequisites

- Python 3.11 (via pyenv), Poetry 1.8+, Docker, Node.js 18+
- AWS account (SageMaker), OpenAI API key, HuggingFace token

### 1. Environment

```bash
git clone https://github.com/sh-arka22/moodSwarm.git && cd moodSwarm
pyenv local 3.11.8
poetry install
cp .env.example .env  # Fill in API keys
```

### 2. Start infrastructure

```bash
docker compose up -d              # MongoDB + Qdrant
poetry run zenml init             # Initialize ZenML
```

### 3. Run data pipelines

```bash
# Crawl content and store in MongoDB
poetry run python -m tools.run --run-etl --no-cache

# Clean, chunk, embed and store in Qdrant
poetry run python -m tools.run --run-feature-engineering --no-cache
```

### 4. Deploy LLM endpoint

```bash
poetry run python -m tools.deploy_endpoint create    # ~5-10 min, ~$1.20/hr
poetry run python -m tools.deploy_endpoint status     # Wait for "InService"
```

### 5. Launch the app

```bash
# Terminal 1 — Backend
poetry run python -m tools.ml_service                 # FastAPI on :8000

# Terminal 2 — Frontend
cd frontend && npm install
npx expo start                                        # Press 'w' for web
```

### 6. Clean up

```bash
poetry run python -m tools.deploy_endpoint delete     # Stop billing
docker compose down                                   # Stop local services
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React Native (Expo), expo-router, Drawer navigation |
| Backend API | FastAPI, Pydantic, Opik tracing |
| Orchestration | ZenML pipelines |
| Data stores | MongoDB (documents), Qdrant (vectors) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-4-v2 |
| RAG | Self-query, query expansion, cross-encoder reranking |
| Fine-tuning | Unsloth QLoRA (SFT + DPO) on Llama 3.1 8B |
| Inference | AWS SageMaker, HuggingFace TGI, INT8 quantization |
| CI/CD | GitHub Actions, Docker, AWS ECR |
| Quality | Ruff, pytest, pre-commit, gitleaks |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/rag` | One-shot RAG query (legacy) |
| `POST` | `/conversations` | Create conversation thread |
| `GET` | `/conversations` | List all threads |
| `PATCH` | `/conversations/{id}` | Rename thread |
| `DELETE` | `/conversations/{id}` | Delete thread + messages |
| `GET` | `/conversations/{id}/messages` | Get message history |
| `POST` | `/conversations/{id}/messages` | Send message, get RAG response |

## Project Structure

```
llm_engineering/          # Core package (Domain-Driven Design)
├── domain/               #   Data models (MongoDB + Qdrant ODMs)
├── application/          #   Crawlers, preprocessing, RAG, embeddings
├── infrastructure/       #   FastAPI, DB connectors, AWS deploy
└── model/                #   Fine-tuning, inference, evaluation

pipelines/                # ZenML pipeline definitions
steps/                    # Reusable pipeline components
tools/                    # CLI utilities (run, deploy, inspect, evaluate)
configs/                  # Pipeline YAML configs
tests/                    # Unit + integration tests
frontend/                 # React Native (Expo) chat UI
docs/                     # Architecture, runbook, weekly reports
```

## Model Lineage

```
unsloth/Meta-Llama-3.1-8B
  └─ SFT (QLoRA, 3 epochs, 399 steps) ─► saha2026/TwinLlama-3.1-8B
       └─ DPO (beta=0.5, 1 epoch) ─► saha2026/TwinLlama-3.1-8B-DPO  ← deployed
```

## Key Metrics

| Metric | Value |
|--------|-------|
| RAG Recall@3 | 0.43 |
| RAG Recall@6 | 0.60 |
| MRR@3 | 1.00 |
| SFT training loss | 1.12 → 0.49 |
| LLM-as-judge accuracy | 2.23 / 3.0 |
| LLM-as-judge style | 2.12 / 3.0 |

## Documentation

| Document | Description |
|----------|-------------|
| [Operations Runbook](docs/RUNBOOK.md) | Deploy, rollback, incident response |
| [Architecture Deep Dive](docs/architecture.md) | Patterns, data flow, design decisions |
| [Development Guide](docs/development.md) | Full command reference, week-by-week build log |
| [MLOps & CI/CD](docs/week8_mlops.md) | CI/CD pipelines, Docker, testing strategy |

## Cost

| Resource | Cost | Notes |
|----------|------|-------|
| SageMaker ml.g5.xlarge | ~$1.20/hr | Only when endpoint is deployed |
| SFT training | ~$4.50 | One-time, ml.g5.2xlarge |
| DPO training | ~$0.60 | One-time, ml.g5.2xlarge |
| OpenAI (RAG pipeline) | ~$0.01/query | gpt-4o-mini for query expansion |
| Local infra | Free | Docker MongoDB + Qdrant |

## License

MIT
