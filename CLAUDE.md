# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoodSwarm is an 8-week solo-builder LLM engineering project following *The LLM Engineer's Handbook* (11 chapters) by Paul Iusztin and Maxime Labonne. The companion reference repo is `sh-arka22/LLM-Engineers-Handbook-Forked`.

**Current state:** Week 1 complete — project skeleton, Docker services, and smoke test pipeline working.

## Architecture (Target)

```
Data Collection → ETL Pipeline → Feature Store → Model Training → Inference & Evaluation
      ↓               ↓               ↓              ↓                    ↓
  External          MongoDB         Qdrant       HuggingFace          FastAPI /rag
  sources        (raw docs)      (vectors)      (SFT/DPO)           AWS SageMaker
```

- **Package layout:** Domain-Driven Design — `llm_engineering/` with domain, application, model, infrastructure layers
- **Pipelines:** ZenML (`pipelines/`, `steps/`, `tools/run.py`)
- **Datastores:** MongoDB (raw documents), Qdrant (vector embeddings)
- **Model lifecycle:** Dataset generation → SFT training → DPO training → evaluation → inference
- **Deployment:** AWS SageMaker endpoint + FastAPI `/rag` endpoint
- **Config:** Environment vars defined in `llm_engineering/settings.py`, pipeline configs in `configs/`

## Prerequisites

**Accounts needed:** HuggingFace, AWS (SageMaker), Comet ML, MongoDB Atlas (optional), Qdrant Cloud (optional)

**Environment variables** (from `llm_engineering/settings.py`, configured in `.env`):
- `DATABASE_HOST` — MongoDB connection string (default: local Docker)
- `QDRANT_DATABASE_HOST`, `QDRANT_DATABASE_PORT` — Qdrant connection (default: localhost:6333)
- `OPENAI_API_KEY` — used for dataset generation and evaluation
- `HUGGINGFACE_ACCESS_TOKEN` — model/dataset hub access
- `COMET_API_KEY` — experiment tracking

**Local services:** Docker (for MongoDB + Qdrant), Python 3.11 (via pyenv), Poetry 1.8+, ZenML 0.74.0

## Common Commands

```bash
# Environment setup
pyenv local 3.11.8                    # Set Python version
poetry install                        # Install dependencies

# Docker services
docker compose up -d                  # Start MongoDB + Qdrant
docker compose down                   # Stop services
docker ps                             # Verify containers running

# Run pipelines
poetry run python -m tools.run --run-smoke-test --no-cache    # Smoke test
poetry run ruff check .               # Lint check
poetry run ruff check . --fix         # Auto-fix lint errors

# ZenML
poetry run zenml status               # Check ZenML stack
poetry run zenml pipeline list        # List pipelines
```

## Key Files

- `instructions.md` — 8-week roadmap with phase objectives, success criteria, and working rules
- `instructionsclaude.md` — Meta-instructions for Claude execution mode and memory management
- `LLM-Engineers-Handbook.pdf` — Reference handbook (11 chapters + appendix)

## Execution Style

Follow the output template from `instructions.md` for every response:

1. **Today Goal** (1 sentence)
2. **Do This Now** (max 5 steps)
3. **Commands** (copy-paste block)
4. **Success Check**
5. **Fast Failure Recovery**
6. **Next Small Step**

Additional rules:
- Be practical over perfect; prioritize shipping over theory
- Prefer smallest useful next step
- For cloud tasks, include expected cost risk and cheaper fallback
- For major choices, output a decision table: Option / Effort (S/M/L) / Risk / Cost / Recommendation
- For claimed completions, include: command run, observed output, artifact path, PASS/FAIL/UNVERIFIED

## Reference Repo Structure

From the companion repo (`sh-arka22/LLM-Engineers-Handbook-Forked`):

- `code_snippets/` — Standalone example code
- `configs/` — Pipeline configuration files
- `llm_engineering/` — Core DDD code (domain, application, model, infrastructure)
- `pipelines/` — ZenML ML pipeline definitions
- `steps/` — Reusable pipeline components
- `tests/` — Sample tests
- `tools/` — Utility scripts (`run.py`, `ml_service.py`, `rag.py`, `data_warehouse.py`)

---

## 8-Week Execution Plan

### Book Chapter → Week Mapping

| Week | Chapters | Phase |
|------|----------|-------|
| 1 | Ch 1 (Architecture), Ch 2 (Tooling) | Setup |
| 2 | Ch 3 (Data Engineering) | ETL |
| 3 | Ch 4 (RAG Feature Pipeline) | Feature Store |
| 4 | Ch 9 (RAG Inference Pipeline) | Retrieval Baseline |
| 5 | Ch 5 (Supervised Fine-Tuning) | SFT Training |
| 6 | Ch 6 (Preference Alignment), Ch 7 (Evaluation) | DPO + Eval |
| 7 | Ch 8 (Inference Optimization), Ch 10 (Deployment) | Deploy |
| 8 | Ch 11 (MLOps/LLMOps), Appendix | Ops + Capstone |

---

### Week 1: Foundation & Environment Setup
**Chapters:** 1 (LLM Twin Architecture), 2 (Tooling & Installation) | **Phase 1A**

**Tasks:**
1. Clone reference repo `sh-arka22/LLM-Engineers-Handbook-Forked`
2. Set up Python env with Poetry: `poetry install --without aws`
3. Configure env vars from `llm_engineering/settings.py`
4. Start Docker services: MongoDB + Qdrant
5. Install and configure ZenML: `zenml init`, register stack
6. Run a smoke test pipeline in mock/dummy mode
7. Produce minimal runbook: setup commands, env vars checklist, common failures

**Math:** FTI pipeline architecture concepts
**Metrics:** All services healthy, ZenML dashboard accessible, one pipeline completes
**Risk:** Docker resource limits; fallback = MongoDB Atlas free tier
**Milestone:** Local dev environment fully operational with all services running

---

### Week 2: Data Collection & ETL Pipeline
**Chapters:** 3 (Data Engineering) | **Phase 1B**

**Tasks:**
1. Study data collection pipeline: dispatcher → GitHubCrawler, CustomArticleCrawler, MediumCrawler
2. Study ODM (Object Document Mapping) pattern for MongoDB
3. Run `digital_data_etl` pipeline end-to-end
4. Harden ETL: add visible success/failure counts per crawl source
5. Verify deduplication — reruns must not create bad duplicates
6. Add/verify retry + exponential backoff for flaky external crawls
7. If crawlers fail, import backed-up data (Ch 3 troubleshooting)

**Math:** Data normalization, text preprocessing statistics
**Metrics:** ETL success rate per source, document counts in MongoDB, zero duplicate records
**Risk:** Selenium issues with dynamic sites; fallback = use backed-up dataset
**Milestone:** ETL runs end-to-end with clear logs and predictable artifacts

---

### Week 3: RAG Feature Pipeline & Vector Store
**Chapters:** 4 (RAG Feature Pipeline) | **Phase 2A**

**Tasks:**
1. Study embeddings theory, vanilla RAG framework, vector DB internals (HNSW)
2. Study clean → chunk → embed pipeline architecture
3. Review chunking handlers and embedding handlers in codebase
4. Validate chunk size/overlap settings by data category (articles vs code vs posts)
5. Run RAG feature pipeline: query warehouse → clean → chunk → embed → load to Qdrant
6. Verify embedded records in Qdrant collections
7. Inspect vector dimensionality and collection stats
8. Implement CDC sync between MongoDB and Qdrant

**Math:** Embedding spaces, cosine similarity, vector indexing (HNSW), matrix factorization
**Metrics:** Qdrant collection sizes, embedding dimensions, chunk count per document category
**Risk:** Memory issues with large embedding batches; fallback = reduce batch size
**Milestone:** All documents cleaned, chunked, embedded, and stored in Qdrant

---

### Week 4: RAG Retrieval & Baseline Quality
**Chapters:** 9 (RAG Inference Pipeline) | **Phase 2B**

**Tasks:**
1. Study advanced RAG: self-query, query expansion, reranking, filtered vector search
2. Implement the retrieval module from codebase
3. Run retrieval tests via RAG inference path
4. Implement query expansion and self-querying pre-retrieval optimizations
5. Implement reranking post-retrieval optimization
6. Establish baseline: Recall@K (minimum), MRR if feasible
7. Record default retrieval params: `k`, expansion count, rerank top-k
8. Build test set of query → expected-context pairs for regression testing

**Math:** Recall@K, MRR, NDCG, Bayes' theorem for relevance, cosine vs dot-product similarity
**Metrics:** Recall@5, Recall@10, MRR, latency per query
**Risk:** Low retrieval quality; fix = tune chunk size, overlap, k, reranker threshold
**Milestone:** Query and retrieve relevant context consistently with measured baselines

---

### Week 5: Instruction Dataset & SFT Training
**Chapters:** 5 (Supervised Fine-Tuning) | **Phase 3A**

**Tasks:**
1. Study instruction datasets, data curation (filtering, dedup, decontamination), SFT techniques
2. Generate instruction dataset using the book's pipeline
3. Apply data quality evaluation: rule-based filtering, deduplication, decontamination
4. Validate sample quality and train/test split statistics
5. Push dataset to HuggingFace only after local checks pass
6. Fine-tune Llama 3.1 8B using QLoRA (most memory-efficient)
7. Track experiment in Comet ML: config, dataset version, loss curves
8. Evaluate SFT model with basic metrics

**Math:** LoRA (low-rank decomposition, rank selection), learning rate schedules, gradient checkpointing, cross-entropy loss
**Metrics:** Training loss, validation loss, perplexity on held-out set
**Risk:** GPU memory limits; fallback = reduce batch size, gradient accumulation, or cloud GPU (~$1-3/hr)
**Milestone:** SFT baseline model trained and checkpointed with tracked experiment

---

### Week 6: Preference Alignment (DPO) & Evaluation
**Chapters:** 6 (Preference Alignment), 7 (Evaluating LLMs) | **Phase 3B**

**Tasks:**
1. Study preference datasets, RLHF vs DPO theory
2. Generate preference dataset (chosen/rejected pairs)
3. Validate preference data quality and statistics
4. Train DPO on top of SFT baseline using Unsloth library
5. Study evaluation methods: general-purpose, domain-specific, task-specific
6. Study RAG evaluation with Ragas and ARES frameworks
7. Run evaluation pipeline: generate answers → evaluate with multiple criteria
8. Compare SFT baseline vs DPO model in experiment log

**Math:** DPO loss (Bradley-Terry model), KL divergence, BLEU/ROUGE, Ragas metrics (faithfulness, relevance, context recall)
**Metrics:** DPO training loss, evaluation scores (BLEU, style similarity), Ragas metrics
**Risk:** DPO can degrade model if preference data is noisy; fix = curate data, reduce epochs
**Milestone:** Reproducible SFT baseline + one DPO-improved candidate with comparative evaluation

---

### Week 7: Inference Optimization & Deployment
**Chapters:** 8 (Inference Optimization), 10 (Deployment) | **Phase 4A**

**Tasks:**
1. Study KV cache, continuous batching, speculative decoding, model parallelism, quantization
2. Quantize best model using GGUF/llama.cpp or GPTQ
3. Benchmark quantized vs full model: quality degradation vs speed gain
4. Study deployment strategies: online, async, batch; monolith vs microservices
5. Deploy inference endpoint to AWS SageMaker using HuggingFace DLCs
6. Configure SageMaker roles and scaling
7. Build FastAPI microservice exposing `/rag` endpoint
8. Validate end-to-end: user query → FastAPI → SageMaker → response
9. Capture latency + failure behavior under load

**Math:** Quantization (INT8/INT4), Flash Attention, throughput vs latency tradeoffs
**Metrics:** Inference latency (p50, p95, p99), throughput (req/sec), model quality post-quantization
**Risk:** SageMaker costs (~$1-5/hr for ml.g5.xlarge); fallback = smaller instance or local inference
**Milestone:** End-to-end RAG inference working via FastAPI + SageMaker with measured latency

---

### Week 8: MLOps, Monitoring & Handover (Capstone)
**Chapters:** 11 (MLOps/LLMOps), Appendix (MLOps Principles) | **Phase 4B**

**Tasks:**
1. Study DevOps → MLOps → LLMOps evolution, CI/CD/CT pipelines
2. Study 6 MLOps principles: automation, versioning, experiment tracking, testing, monitoring, reproducibility
3. Deploy all pipelines to cloud (MongoDB Atlas/Docker, Qdrant Cloud/Docker, ZenML cloud)
4. Containerize code using Docker
5. Set up CI/CD pipeline with GitHub Actions (lint → test → deploy)
6. Add prompt monitoring layer (Opik) on inference pipeline
7. Set up alerting for model quality degradation
8. Run final evaluation pipeline comparing all model candidates
9. Write deploy/rollback checklist
10. Prepare solo operations docs: weekly maintenance + incident response steps

**Math:** Data drift detection (KL divergence for distribution shift), A/B testing fundamentals
**Metrics:** CI/CD pass rate, monitoring alert latency, full system end-to-end latency
**Risk:** Cloud costs accumulating; set billing alerts, tear down unused resources after testing
**Milestone:** End-to-end system operable by one person with clear runbooks, CI/CD, and monitoring
