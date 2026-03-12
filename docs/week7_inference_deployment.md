# Week 7: Inference Optimization & Deployment

## Overview

Deployed the DPO-aligned model (`saha2026/TwinLlama-3.1-8B-DPO`) as a SageMaker real-time inference endpoint and built a FastAPI `/rag` endpoint that chains RAG retrieval with LLM generation.

## Architecture

```
User Query → FastAPI /rag → ContextRetriever (RAG) → SageMaker Endpoint → Response
                                    ↓
                            SelfQuery → QueryExpansion → Qdrant Search → Reranker
```

## Deployment Configuration

| Parameter | Value |
|-----------|-------|
| Model | `saha2026/TwinLlama-3.1-8B-DPO` |
| Instance Type | `ml.g5.2xlarge` (1x NVIDIA A10G, 24GB VRAM) |
| Quantization | bitsandbytes INT8 |
| Container | HuggingFace TGI v2.2.0 |
| Max Input Length | 2048 tokens |
| Max Total Tokens | 4096 tokens |
| Max New Tokens | 150 |
| Temperature | 0.01 |
| Top P | 0.9 |

## Cost Analysis

| Item | Cost |
|------|------|
| SageMaker endpoint (running) | ~$1.62/hr |
| Per day (if left running) | ~$38.88/day |
| Endpoint creation | Free (pay for instance time) |

**Critical:** Always delete the endpoint when done testing!

## Commands

```bash
# Deploy endpoint (~5-15 min startup)
poetry run python -m tools.deploy_endpoint create

# Check endpoint status
poetry run python -m tools.deploy_endpoint status

# Start FastAPI server
poetry run python -m tools.ml_service

# Test RAG endpoint
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "How do RAG systems work?"}'

# Delete endpoint (stops billing!)
poetry run python -m tools.deploy_endpoint delete
```

## Monitoring

Opik (powered by Comet ML) tracks:
- Model ID and embedding model ID per request
- Temperature and generation parameters
- Token counts: query, context, and answer
- Tagged as "rag" for filtering

## Files Created

| File | Purpose |
|------|---------|
| `llm_engineering/domain/inference.py` | ABC: Inference, DeploymentStrategy |
| `llm_engineering/model/inference/` | SageMaker endpoint client + InferenceExecutor |
| `llm_engineering/model/utils.py` | ResourceManager for endpoint lifecycle |
| `llm_engineering/infrastructure/aws/deploy/` | SageMaker deployment infrastructure |
| `llm_engineering/infrastructure/opik_utils.py` | Opik/Comet monitoring setup |
| `llm_engineering/infrastructure/inference_pipeline_api.py` | FastAPI `/rag` endpoint |
| `tools/ml_service.py` | FastAPI uvicorn launcher |
| `tools/deploy_endpoint.py` | Deploy/delete/status CLI |
