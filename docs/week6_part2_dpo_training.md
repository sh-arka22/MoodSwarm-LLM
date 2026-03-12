# Week 6 Part 2 — DPO Training + SFT vs DPO Evaluation

## DPO Training

### Configuration
```yaml
# configs/training.yaml
parameters:
  finetuning_type: dpo
  num_train_epochs: 1
  per_device_train_batch_size: 2
  learning_rate: 2e-6
  dataset_huggingface_workspace: saha2026
  is_dummy: false
```

### Training Details
| Parameter | Value |
|-----------|-------|
| Base model | `saha2026/TwinLlama-3.1-8B` (SFT checkpoint) |
| Output model | `saha2026/TwinLlama-3.1-8B-DPO` |
| Dataset | `saha2026/llmtwin-dpo` (71 train / 9 test) |
| DPO beta | 0.5 |
| Learning rate | 2e-6 (150x lower than SFT) |
| Epochs | 1 |
| Batch size | 2 (gradient accumulation: 8) |
| Optimizer | AdamW 8-bit |
| Precision | bf16 |
| QLoRA rank | 32, alpha=32, targets: q/k/v/o/gate/down/up_proj |
| Instance | ml.g5.2xlarge (24GB VRAM) |
| Max sequence length | 1024 (prompt) + 1024 (response) |

### Results
| Metric | Value |
|--------|-------|
| Training time | 27.2 seconds |
| Total wall time | 23m 43s (incl. setup, deps, model upload) |
| Billable seconds | 1324 (~$0.60 at $1.62/hr) |
| DPO loss | 0.6931 -> 0.7011 |
| Rewards accuracy | 0.0 -> 0.5 |
| Chosen logps | -132.76 to -102.88 |
| Rejected logps | -43.33 to -34.90 |
| Chosen rewards | -0.009 to +0.004 |
| Rejected rewards | -0.002 to +0.009 |

### Comet ML Experiment
- Name: `selective_heel_8570`
- URL: https://www.comet.com/sh-arka22/twin/a54c94dbc1284881a5f0317fbf28be8a

### Inference Test (on SageMaker)
Prompt: "Write a paragraph to introduce supervised fine-tuning."

Output: "Supervised fine-tuning is a method used to enhance the performance of a pre-trained machine learning model by adjusting its parameters based on a labeled dataset. In this approach, the model is initialized with weights obtained from a larger dataset, which provides a strong baseline for the task at hand. The fine-tuning process involves retraining the model on a smaller dataset, allowing it to adapt to the specific requirements of the new task. This approach can lead to significant improvements in accuracy and performance, as the model leverages its existing knowledge while refining its parameters to better fit the new data."

## SFT vs DPO Comparison

### Proxy Evaluation (LLM-as-Judge)
Since we can't run inference locally (no GPU), we compared dataset answers as a proxy:

| Metric | SFT Baseline | DPO Chosen | DPO Rejected |
|--------|:------------:|:----------:|:------------:|
| Accuracy (avg) | 2.23 | 2.05 | 1.80 |
| Style (avg) | 2.12 | 1.95 | 1.85 |
| Samples evaluated | 65 | 20 | 20 |

### Score Distributions
| Category | Score 1 | Score 2 | Score 3 |
|----------|:-------:|:-------:|:-------:|
| SFT Accuracy | 1 | 48 | 16 |
| SFT Style | 0 | 57 | 8 |
| DPO Chosen Accuracy | 0 | 19 | 1 |
| DPO Chosen Style | 1 | 19 | 0 |
| DPO Rejected Accuracy | 4 | 16 | 0 |
| DPO Rejected Style | 3 | 17 | 0 |

### Key Insights
- **Chosen > Rejected on style** by 0.10 points (PASS) — preference signal is valid
- **Chosen > Rejected on accuracy** by 0.25 points — chosen answers are more factually correct
- **SFT baseline scores higher** than DPO chosen (2.12 vs 1.95 style) — expected since SFT dataset was generated with higher-quality prompts
- DPO training should close this gap by learning to prefer chosen-style answers

### Cost Summary
| Item | Cost |
|------|------|
| SageMaker training (22 min) | ~$0.60 |
| LLM-as-judge evaluation (40 samples) | ~$0.01 |
| **Total** | **~$0.61** |

## Bug Fixed
- `steps/training/train.py` imported `llm_engineering.model.finetuning.sagemaker` but the actual file is `sagemaker_launcher.py`. Fixed import path. Note: evaluation module uses `sagemaker.py` while finetuning uses `sagemaker_launcher.py` — naming inconsistency.

## Artifacts
- Model: https://huggingface.co/saha2026/TwinLlama-3.1-8B-DPO
- Dataset: https://huggingface.co/datasets/saha2026/llmtwin-dpo
- Comparison data: `data/sft_vs_dpo_comparison.json`
- Comparison tool: `tools/model_compare.py`
