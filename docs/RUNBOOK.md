# MoodSwarm Operations Runbook

> Solo-builder reference for operating the MoodSwarm LLM Twin platform.

---

## 1. Daily Operations

### Health Checks

```bash
# Local services
docker ps                                              # MongoDB + Qdrant running
poetry run zenml status                                # ZenML stack healthy
poetry run python -m tools.qdrant_inspect list-collections  # Qdrant collections present

# SageMaker endpoint (only if deployed)
poetry run python -m tools.deploy_endpoint status      # Check endpoint InService
```

### Monitoring

- **Comet ML dashboard:** Check Opik traces for inference latency, token counts, error rates
- **AWS Console → SageMaker → Endpoints:** Check endpoint health, invocation metrics
- **CloudWatch (if configured):** ModelLatency, Invocations, 4xx/5xx errors

---

## 2. Weekly Maintenance

| Task | Command | Notes |
|------|---------|-------|
| Dependency audit | `poetry update --dry-run` | Review before updating |
| Lint check | `poetry poe lint-check` | Fix with `poetry poe lint-fix` |
| Run test suite | `poetry poe test` | All tests must pass |
| Model evaluation | `poetry run python -m tools.model_compare --n 20` | Track quality drift |
| Cost review | AWS Console → Billing | Check SageMaker, ECR, S3 costs |
| Qdrant stats | `poetry run python -m tools.qdrant_inspect collection-stats embedded_articles` | Monitor collection sizes |

---

## 3. Deployment Checklist

### Pre-flight: Deploy SageMaker Endpoint

1. [ ] Verify AWS credentials: `aws sts get-caller-identity`
2. [ ] Confirm model exists on HuggingFace: `saha2026/TwinLlama-3.1-8B-DPO`
3. [ ] Check quota: AWS Console → Service Quotas → SageMaker → `ml.g5.xlarge`
4. [ ] Estimate cost: ~$1.20/hr for `ml.g5.xlarge`
5. [ ] Deploy: `poetry run python -m tools.deploy_endpoint create`
6. [ ] Wait for InService: `poetry run python -m tools.deploy_endpoint status`
7. [ ] Smoke test: `curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" -d '{"query": "test"}'`
8. [ ] Set reminder to delete endpoint when done

### Post-deploy

```bash
# Start FastAPI server
poetry run python -m tools.ml_service

# Test the /rag endpoint
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "How do RAG systems work?"}'
```

---

## 4. Rollback Procedures

### Endpoint Rollback

```bash
# Delete current endpoint
poetry run python -m tools.deploy_endpoint delete

# Redeploy with previous model (edit settings.py HF_MODEL_ID first)
# e.g., rollback from DPO to SFT:
#   HF_MODEL_ID = "saha2026/TwinLlama-3.1-8B"
poetry run python -m tools.deploy_endpoint create
```

### Model Rollback

Models are versioned on HuggingFace:
- **SFT baseline:** `saha2026/TwinLlama-3.1-8B`
- **DPO (current):** `saha2026/TwinLlama-3.1-8B-DPO`
- **Base model:** `unsloth/Meta-Llama-3.1-8B`

Change `HF_MODEL_ID` in `.env` or `settings.py` and redeploy.

### Data Rollback

```bash
# Re-import MongoDB data from backup
poetry run python -m tools.data_warehouse --import-raw-data

# Re-run feature engineering to rebuild Qdrant
poetry poe run-feature-engineering
```

---

## 5. Incident Response

### Endpoint Not Responding

1. Check status: `poetry run python -m tools.deploy_endpoint status`
2. Check CloudWatch logs: AWS Console → CloudWatch → Log groups → `/aws/sagemaker/Endpoints/twin`
3. If `Failed`: delete and recreate endpoint
4. If quota exceeded: request limit increase or try different region

### Empty Responses from LLM

The model requires **Alpaca template** (`### Instruction:` / `### Response:`). If responses are empty:
1. Verify `InferenceExecutor` prompt has Alpaca markers
2. Check `return_full_text: False` in payload
3. The executor has automatic fallback: retries with `return_full_text: True`

### OOM (Out of Memory) on Endpoint

1. Reduce `MAX_INPUT_LENGTH` in settings (current: 3072)
2. Reduce `MAX_TOTAL_TOKENS` (current: 4096)
3. Ensure INT8 quantization is enabled in deploy config
4. If persistent: upgrade to `ml.g5.2xlarge` (~$2.40/hr)

### Qdrant Connection Failure

```bash
# Check Docker
docker ps | grep qdrant
docker compose up -d  # Restart if needed

# Verify connection
poetry run python -m tools.qdrant_inspect list-collections
```

### MongoDB Connection Failure

```bash
docker ps | grep mongo
docker compose up -d
# Test: poetry run python -m tools.data_warehouse --export-raw-data
```

---

## 6. Cost Management

### Active Cost Items

| Resource | Cost | When Active |
|----------|------|-------------|
| SageMaker `ml.g5.xlarge` | ~$1.20/hr | Only when endpoint deployed |
| ECR storage | ~$0.10/GB/month | Always (if images pushed) |
| MongoDB Atlas (if used) | Free tier | Always |
| Qdrant Cloud (if used) | Free tier | Always |

### Cost Safety

```bash
# ALWAYS delete endpoint when done testing
poetry run python -m tools.deploy_endpoint delete

# Verify deletion
poetry run python -m tools.deploy_endpoint status
# Should show: endpoint not found
```

### Billing Alerts

Set up AWS Budget alarm for $10/day:
- AWS Console → Billing → Budgets → Create budget
- Threshold: $10/day, notify via email

---

## 7. Common Commands Reference

```bash
# === Infrastructure ===
docker compose up -d                    # Start MongoDB + Qdrant
docker compose down                     # Stop services
poetry run zenml status                 # ZenML health

# === Pipelines ===
poetry poe run-smoke-test               # Health check
poetry poe run-digital-data-etl         # ETL pipeline
poetry poe run-feature-engineering      # Feature engineering
poetry poe run-training                 # SFT/DPO training
poetry poe run-evaluation               # Model evaluation

# === Inference ===
poetry poe deploy-endpoint              # Deploy SageMaker endpoint
poetry poe endpoint-status              # Check endpoint status
poetry poe delete-endpoint              # Delete endpoint (stops billing)
poetry poe start-api                    # Start FastAPI server

# === Quality ===
poetry poe lint-check                   # Ruff lint
poetry poe format-check                 # Ruff format
poetry poe test                         # Run pytest suite

# === Data ===
poetry poe export-data-warehouse        # MongoDB → JSON
poetry poe import-data-warehouse        # JSON → MongoDB

# === Tools ===
poetry run python -m tools.rag -q "query" --k 3       # RAG retrieval
poetry run python -m tools.model_compare --n 20        # Model comparison
poetry run python -m tools.dataset_inspect stats       # Dataset stats
```
