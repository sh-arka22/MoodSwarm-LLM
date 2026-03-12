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
| Instance Type | `ml.g5.xlarge` (1x NVIDIA A10G, 24GB VRAM, 4 vCPUs, 16GB RAM) |
| Quantization | bitsandbytes INT8 |
| Container | HuggingFace TGI v2.4.0 |
| Max Input Length | 3,072 tokens |
| Max Total Tokens | 4,096 tokens |
| Max Batch Total Tokens | 4,096 tokens |
| Max New Tokens | 150 |
| Temperature | 0.01 |
| Top P | 0.9 |
| Prompt Template | Alpaca format (`### Instruction:` / `### Response:`) |
| Health Check Timeout | 900s |

## Prompt Template

The model was SFT-trained with the Alpaca template, so inference prompts **must** use the same format:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}

### Response:
```

Sending freeform prompts (without the Alpaca wrapper) causes the model to immediately emit EOS with zero generated tokens.

## Cost Analysis

| Item | Cost |
|------|------|
| SageMaker endpoint (running) | ~$1.20/hr |
| Per day (if left running) | ~$28.80/day |
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

## Deployment Issues & Fixes

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `ValueError: Must setup local AWS configuration with a region` | `get_huggingface_llm_image_uri()` creates a default SageMaker Session without region | Created explicit `boto3.Session(region_name=...)` → `sagemaker.Session(boto_session=...)` and passed through entire deploy chain |
| 2 | `ResourceLimitExceeded` for ml.g5.2xlarge | AWS account had 0 quota for ml.g5.2xlarge **endpoint** usage (separate from training quota) | Switched to `ml.g5.xlarge` (same A10G GPU, quota=1) |
| 3 | `data did not match any variant of untagged enum ModelWrapper` | TGI v2.2.0's bundled `tokenizers` library can't parse Llama 3.1's `tokenizer.json` format | Upgraded TGI image from v2.2.0 to v2.4.0 |
| 4 | `CUDA error: an illegal memory access` / CUDA OOM | `MAX_TOTAL_TOKENS=8192` required too much KV cache for A10G with INT8 | Set `MAX_INPUT_LENGTH=3072`, `MAX_TOTAL_TOKENS=4096`, `MAX_BATCH_TOTAL_TOKENS=4096` |
| 5 | Empty `generated_text` (zero tokens) | Model SFT'd with Alpaca template but RAG prompt was freeform — model hit EOS immediately | Wrapped prompt in Alpaca template; added fallback: retry with `return_full_text: True` and strip input |
| 6 | `AttributeError: ResourceNotFoundException` in `endpoint_exists()` | boto3 SageMaker uses `ClientError`, not `ResourceNotFoundException` | Changed exception handler to catch `ClientError` |

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
