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

Deployment required **5 iterations** before the endpoint was fully operational. Each issue is documented below with the full error, root cause analysis, what was tried, and the final fix.

---

### Issue #1: AWS Region Not Configured

**Error:**
```
ValueError: Must setup local AWS configuration with a region supported by SageMaker.
```

**Where:** `create_endpoint()` in `llm_engineering/infrastructure/aws/deploy/run.py`

**Root Cause:** The `get_huggingface_llm_image_uri()` function from the SageMaker SDK internally creates a default `sagemaker.Session()`. This default session tries to read the AWS region from `~/.aws/config`, which wasn't configured on this machine. Even though we had `AWS_REGION` set in `.env` and used it for boto3 clients, the SageMaker SDK doesn't read from environment variables by default.

**Fix:** Created an explicit boto3 session with the region and passed it through the entire deployment chain:
```python
# run.py — create_endpoint()
boto_session = boto3.Session(
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY,
    aws_secret_access_key=settings.AWS_SECRET_KEY,
)
sagemaker_session = Session(boto_session=boto_session)
llm_image = get_huggingface_llm_image_uri("huggingface", version="2.4.0", session=sagemaker_session)
```

Had to also add `sagemaker_session` parameter to `SagemakerHuggingfaceStrategy.deploy()`, `DeploymentService.deploy()`, `prepare_and_deploy_model()`, and `HuggingFaceModel(sagemaker_session=...)`.

**Files Changed:** `run.py`, `sagemaker_huggingface.py` (both in `aws/deploy/`)

**Lesson:** Always pass explicit sessions when using AWS SDKs programmatically. Don't rely on `~/.aws/config` — it may not exist in all environments.

---

### Issue #2: SageMaker Endpoint Quota Limit

**Error:**
```
botocore.exceptions.ClientError: An error occurred (ResourceLimitExceeded) when calling the
CreateEndpoint operation: The account-level service limit 'ml.g5.2xlarge for endpoint usage'
is 0 Instances, with current utilization of 0 Instances and a request delta of 1 Instances.
```

**Where:** `HuggingFaceModel.deploy()` inside `prepare_and_deploy_model()`

**Root Cause:** AWS has **separate quotas** for training instances and endpoint (inference) instances. We had quota for `ml.g5.2xlarge` for training jobs (used in Weeks 5-6), but **zero quota** for the same instance type for real-time endpoints. These are tracked under different Service Quotas:
- Training: `ml.g5.2xlarge for training job usage` → had quota
- Endpoint: `ml.g5.2xlarge for endpoint usage` → quota = 0

**What Was Tried:** Checked quotas programmatically via AWS Service Quotas API:
```python
import boto3
client = boto3.client('service-quotas', region_name='eu-central-1')
# Listed all SageMaker quotas containing "g5" and "endpoint"
```
Found that `ml.g5.xlarge for endpoint usage` had quota = 1.

**Fix:** Changed `GPU_INSTANCE_TYPE` from `ml.g5.2xlarge` to `ml.g5.xlarge` in `settings.py`. Both use the same NVIDIA A10G GPU (24GB VRAM), but `ml.g5.xlarge` has fewer CPUs (4 vs 8) and less RAM (16GB vs 32GB) — sufficient for inference.

**Files Changed:** `llm_engineering/settings.py`

**Lesson:** Always verify endpoint-specific quotas before deploying. Training quotas ≠ endpoint quotas. Use `aws service-quotas list-service-quotas --service-code sagemaker` to check. Request quota increases early — they can take hours to days.

---

### Issue #3: TGI Tokenizer Incompatibility

**Error:**
```
data did not match any variant of untagged enum ModelWrapper at line 1 column 1
```
(Endpoint went to `Failed` status after container startup)

**Where:** Inside the HuggingFace TGI container during model loading

**Root Cause:** TGI v2.2.0 bundles an older version of the `tokenizers` Rust library that cannot parse the `tokenizer.json` format used by Llama 3.1 models. The Llama 3.1 tokenizer uses a newer schema that was only supported starting from `tokenizers` v0.19+, which is bundled in TGI v2.3.0+.

**What Was Tried First:** The initial deploy config used `version="2.2.0"` in `get_huggingface_llm_image_uri()` because the reference book used this version.

**Fix:** Upgraded TGI image version from `2.2.0` to `2.4.0`:
```python
llm_image = get_huggingface_llm_image_uri("huggingface", version="2.4.0", session=sagemaker_session)
```

**Files Changed:** `llm_engineering/infrastructure/aws/deploy/run.py`

**Lesson:** When using newer models (Llama 3.1+), always use TGI v2.3.0 or later. The tokenizer format changed significantly, and older TGI versions silently fail with cryptic Rust-level errors. Check the TGI release notes for model compatibility.

---

### Issue #4: CUDA Out-of-Memory on KV Cache Allocation

**Error:**
```
CUDA error: an illegal memory access was encountered
torch.OutOfMemoryError: CUDA out of memory
```
(Endpoint went to `Failed` after loading the model)

**Where:** Inside the TGI container during KV cache pre-allocation

**Root Cause:** TGI pre-allocates KV cache memory at startup based on `MAX_TOTAL_TOKENS`. With `MAX_TOTAL_TOKENS=8192` and bitsandbytes INT8 quantization, the memory budget was:
- INT8 model weights: ~8GB
- KV cache for 8192 tokens: ~8-10GB
- TGI overhead: ~2-3GB
- Total: ~18-21GB on a 24GB A10G → OOM

The initial attempt used `MAX_INPUT_LENGTH=4096, MAX_TOTAL_TOKENS=8192` thinking it would give more room for long contexts.

**What Was Tried:**
1. `MAX_INPUT_LENGTH=4096, MAX_TOTAL_TOKENS=8192` → OOM
2. `MAX_INPUT_LENGTH=2048, MAX_TOTAL_TOKENS=4096` → worked but too conservative on input

**Fix:** Set the token limits to fit within 24GB:
```python
MAX_INPUT_LENGTH = 3072    # Room for RAG context + query
MAX_TOTAL_TOKENS = 4096    # Input + output combined
MAX_BATCH_TOTAL_TOKENS = 4096  # Single-request batching
```

**Files Changed:** `llm_engineering/settings.py`, `llm_engineering/infrastructure/aws/deploy/config.py`

**Lesson:** For 8B parameter models with INT8 quantization on A10G (24GB):
- `MAX_TOTAL_TOKENS=4096` is safe
- `MAX_TOTAL_TOKENS=8192` will OOM
- Each doubling of `MAX_TOTAL_TOKENS` roughly doubles KV cache memory
- Use `MAX_BATCH_TOTAL_TOKENS = MAX_TOTAL_TOKENS` for single-user testing to minimize memory

**Note:** A stale endpoint config was left behind after the failed deployment. Had to manually delete via boto3:
```python
sagemaker_client.delete_endpoint_config(EndpointConfigName="twin")
sagemaker_client.delete_model(ModelName="huggingface-pytorch-tgi-inference-...")
```

---

### Issue #5: Empty Response — Wrong Prompt Format

**Error:**
```json
{"generated_text": ""}
```
(Endpoint was `InService` and responded, but returned zero generated tokens)

**Where:** `InferenceExecutor.execute()` in `llm_engineering/model/inference/run.py`

**Root Cause:** The model was SFT-trained using the **Alpaca template format**:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}
```

But the initial `InferenceExecutor` sent a freeform prompt like:
```
You are a content creator. Write what the user asked you to...
```

Without the `### Instruction:` / `### Response:` markers, the model didn't recognize the prompt as an instruction and immediately emitted the EOS token, producing zero generated tokens.

**What Was Tried:**
1. Freeform prompt → empty response
2. Checked if `return_full_text: False` was stripping the answer → no, genuinely zero tokens
3. Tested with Alpaca-wrapped prompt → worked immediately

**Fix:** Wrapped the RAG prompt in the Alpaca template:
```python
self.prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}

### Response:
"""
```

Also added a fallback mechanism: if `return_full_text: False` returns empty, retry with `return_full_text: True` and strip the input prefix from the response.

**Files Changed:** `llm_engineering/model/inference/run.py`

**Lesson:** **The inference prompt format MUST match the training prompt format.** This is the single most common failure mode in fine-tuned model deployment. Always check:
1. What template was used during SFT training
2. Whether special tokens (BOS/EOS) are expected
3. Whether the model was trained with a system prompt or not

For Alpaca-format models, never send freeform prompts — always wrap in `### Instruction:` / `### Response:`.

---

### Issue #6: Wrong Exception Type in boto3

**Error:**
```
AttributeError: ResourceNotFoundException
```
(When checking if an endpoint exists using `ResourceManager`)

**Where:** `endpoint_exists()` in `llm_engineering/model/utils.py`

**Root Cause:** The code was written to catch `ResourceNotFoundException`, which is not a valid exception in the boto3 SageMaker client. boto3 raises `botocore.exceptions.ClientError` for all AWS API errors, including "resource not found" scenarios. The specific error code is embedded in the `ClientError` response metadata, not as a separate exception class.

**Fix:** Changed the exception handler from `ResourceNotFoundException` to `ClientError`:
```python
from botocore.exceptions import ClientError

def endpoint_exists(self, endpoint_name: str) -> bool:
    try:
        self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError:
        return False
```

Same fix applied to `endpoint_config_exists()`.

**Files Changed:** `llm_engineering/model/utils.py`

**Lesson:** boto3 does NOT have service-specific exception classes like `ResourceNotFoundException`. All errors come as `ClientError` with an error code in `e.response['Error']['Code']`. If you need to distinguish between "not found" and other errors, check the code:
```python
except ClientError as e:
    if e.response['Error']['Code'] == 'ValidationException':
        return False  # Resource not found
    raise  # Re-raise unexpected errors
```

---

### Summary of All Configuration Changes During Deployment

| Setting | Initial Value | Final Value | Reason |
|---------|--------------|-------------|--------|
| `GPU_INSTANCE_TYPE` | `ml.g5.2xlarge` | `ml.g5.xlarge` | Quota limit (Issue #2) |
| TGI image version | `2.2.0` | `2.4.0` | Tokenizer compat (Issue #3) |
| `MAX_INPUT_LENGTH` | `2048` | `3072` | Tuned after OOM (Issue #4) |
| `MAX_TOTAL_TOKENS` | `4096` → tried `8192` | `4096` | OOM at 8192 (Issue #4) |
| `MAX_BATCH_TOTAL_TOKENS` | `4096` → tried `8192` | `4096` | OOM at 8192 (Issue #4) |
| Prompt format | Freeform | Alpaca template | Zero tokens without it (Issue #5) |
| Exception handler | `ResourceNotFoundException` | `ClientError` | Wrong exception type (Issue #6) |
| SageMaker Session | Default (no region) | Explicit boto3.Session | Region error (Issue #1) |
| Cost estimate | ~$1.62/hr | ~$1.20/hr | Smaller instance (Issue #2) |

## End-to-End Test Results (2026-03-12)

| Step | Result |
|------|--------|
| Endpoint deploy | ~8 minutes cold start → `InService` |
| Query: "How do RAG systems work?" | 150-token answer about encoder-decoder architecture and attention mechanisms |
| Query: "What is supervised fine-tuning?" | Context-grounded answer referencing project content (Mistral, QLORA, Comet ML) |
| Endpoint teardown | Deleted endpoint + config + model in 2 seconds |
| Test cost | ~$0.20 (~10 minutes on ml.g5.xlarge @ $1.20/hr) |

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
