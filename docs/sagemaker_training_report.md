# 🚀 SageMaker Fine-Tuning Journey: From Local Data to TwinLlama

This document provides a comprehensive post-mortem and explanatory guide on how the MoodSwarm TwinLlama model was trained on AWS SageMaker. It covers the end-to-end data flow, infrastructure setup, dependency hell resolution, and the training telemetry.

---

## 1. How Local Data Reached SageMaker

SageMaker instances run in the cloud and do not have direct access to your local MongoDB or Qdrant vector databases. To bridge this gap, we implemented a data staging approach via the Hugging Face Hub:

1. **Local Generation (`tools/push_dataset.py`)**: We extracted the cleaned article chunks, repositories, and posts from the local databases, paired them with synthetic instructions (using GPT-4o-mini locally), and structured them into an Alpaca-format dataset (`instruction`, `output`).
2. **Uploading to Hugging Face**: The dataset was pushed to your Hugging Face workspace (`saha2026/llmtwin`).
3. **Fetching in SageMaker**: Inside the `finetune.py` script running on the AWS GPU instance, we used the `datasets.load_dataset("saha2026/llmtwin")` API to fetch this local knowledge base into the training context seamlessly. We also merged it with the 10k-sample `mlabonne/FineTome-Alpaca-100k` dataset to ensure the model retained its general instruction-following capabilities alongside your persona.

---

## 2. Deploying and Training on AWS SageMaker

The training orchestrator consists of two main pieces:

### A. The Launcher (`sagemaker_launcher.py`)
This script runs locally on your machine and uses the AWS `boto3`/`sagemaker` SDKs to request a spot on an AWS GPU instance.
- **Instance Type**: `ml.g5.2xlarge` (1x NVIDIA A10G 24GB VRAM).
- **Environment**: It packages the entire `llm_engineering/model/finetuning/` folder (including `finetune.py` and `requirements.txt`) and uploads it to an S3 bucket.
- **Trigger**: It tells SageMaker to spin up the instance, download the code from S3, pip install the `requirements.txt`, and execute `finetune.py`.

### B. The Training Script (`finetune.py`)
This script executes inside the remote SageMaker instance:
- **Unsloth FastLanguageModel**: Loads `unsloth/Meta-Llama-3.1-8B` in 4-bit precision to fit within the 24GB VRAM limit.
- **QLoRA Setup**: Attaches Low-Rank Adapters (LoRA) to 7 attention and MLP projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank 32.
- **Training**: Uses Hugging Face's `SFTTrainer` with gradient accumulation, packing, and 8-bit AdamW optimizer.
- **Output**: After 3 epochs, the adapted weights are merged into the base model in 16-bit precision and pushed directly to your Hugging Face Hub as `saha2026/TwinLlama-3.1-8B`. At the end, an `inference()` test is run to sanity-check the model on the instance.

---

## 3. Dependency Hell: Errors Encountered & Solutions

The biggest challenge in this deployment was getting the precise combination of `torch`, `transformers`, and `unsloth` to compile custom GPU kernels correctly without breaking each other. Here is the chronological debug log:

### Error 1: AWS Service Quota Exceeded
- **Issue**: SageMaker threw a `ResourceLimitExceeded` error stating the account had a limit of 0 instances for `ml.g5.2xlarge` for training.
- **Fix**: We navigated the AWS Service Quotas console and submitted a quota increase request for SageMaker training instances. AWS approved it a few hours later.

### Error 2: Unsloth Model ID Parsing Failure
- **Issue**: Unsloth threw a KeyError when trying to parse the hardcoded `meta-llama/Llama-3.1-8B` configurations.
- **Fix**: Unsloth relies on pre-quantized, optimized weights mapped to specific repo names. We corrected the ID to `unsloth/Meta-Llama-3.1-8B` in `finetune.py`.

### Error 3: PyTorch & Torchvision Mismatch
- **Issue**: The SageMaker PyTorch 2.1 base image clashed with our `requirements.txt`. The `pip install` solver failed because Unsloth needs `torch>=2.4.0`, but the default `torchvision` was incompatible.
- **Fix**: We strictly pinned `torch==2.4.0` alongside `torchvision==0.19.0` to force a clean dependency tree.

### Error 4: Safetensors `ModelWrapper` Parsing Error
- **Issue**: During model loading, `transformers` threw `Exception: data did not match any variant of untagged enum ModelWrapper`.
- **Fix**: The book's codebase originally pinned `transformers==4.43.3`. This older version relies on a `tokenizers` library that cannot parse the modern tokenizer structure used by Llama 3.1. We relaxed the pin to `transformers>=4.45.0`.

### Error 5: `PreTrainedConfig` NameError & `torch._inductor` Crash
- **Issue**: Upgrading Transformers to `4.45.2` caused it to rename `PreTrainedConfig` (capital T) to `PretrainedConfig` (lowercase T), which immediately broke the Unsloth version we were using (`2024.9.post2`). 
- **Wait, why not upgrade Unsloth?** When we upgraded Unsloth to `2024.11.8` and `unsloth_zoo` to the latest version, `unsloth_zoo` automatically pulled in `torchao` (Torch Architecture Optimization), which hard-depended on `torch>=2.6.0`, breaking our compilation setup.
- **Fix**: The magic triangle was found! We locked the triplet:
  - `transformers==4.45.2` (Modern enough for Llama 3.1 tokenizers).
  - `unsloth==2024.9.post2` (Stable version built for Torch 2.4).
  - Added `import torch._dynamo` before `from unsloth` in `finetune.py`. This monkey-patched a nasty Inductor compilation bug in PyTorch 2.4 when compiled kernels clashed.

### Error 6: `accelerate` Pip Conflict
- **Issue**: Final check! Pip silently failed and printed an empty `InstallRequirementsError`.
- **Fix**: `unsloth_zoo` strictly required `accelerate>=0.34.1`, but our requirements pinned it to `0.33.0`. We relaxed the pin, creating a fully valid dependency graph.

### Error 7: HuggingFace Generation `cache_implementation` Error
- **Issue**: During the final `inference()` sanity check post-training, Transformers threw an era saying `'dynamic'` is not a valid cache implementation.
- **Fix**: `unsloth` 2024.9 forcibly injected a new caching standard that older Transformers versions rejected. We monkey-patched `ALL_CACHE_IMPLEMENTATIONS.append("dynamic")` in `finetune.py` right before inference to fake validate the configuration.

---

## 4. Conclusion
By resolving these deep dependency conflicts, we mapped your local RAG data into a format SageMaker could digest, efficiently ran QLoRA using the `unsloth_zoo` toolkit to cut training time to **~18 minutes (1083 seconds)** on an A10G, and pushed a fully aligned `TwinLlama` artifact to your Hugging Face Hub, verifiable by Comet ML metrics.
