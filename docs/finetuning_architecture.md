# 🧠 MoodSwarm Fine-Tuning Architecture

This document breaks down the internal architecture of the `finetune.py` script. It explains how the model was trained, what model was used, and the step-by-step data flow from raw instructions to a fully adapted LoRA model.

---

## 1. What Model Was Trained?
We fine-tuned **Meta Llama 3.1 8B** (`unsloth/Meta-Llama-3.1-8B`).
- **Why Unsloth?**: We used the `unsloth` repository version instead of `meta-llama` because Unsloth provides heavily optimized, pre-quantized (4-bit) weights. This allows a 8-Billion parameter model to be fine-tuned on a single 24GB VRAM GPU (like the `ml.g5.2xlarge` A10G instance we used) using **QLoRA** (Quantized Low-Rank Adaptation).

---

## 2. The High-Level Fine-Tuning Flow

The script supports two distinct phases of LLM training: **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)**.

```mermaid
flowchart TD
    subgraph Data Preparation
        SFT_DATA[("Instruction Datasets\n(saha2026/llmtwin +\nFineTome-Alpaca)")]
        DPO_DATA[("Preference Datasets\n(saha2026/llmtwin-dpo)")]
    end

    subgraph AWS SageMaker GPU
        BASE_MODEL(("Base Model\nUnsloth Llama 3.1 8B\n(4-bit Quantized)"))
        
        SFT_TRAINING["SFTTrainer\n(Learn the Persona)"]
        DPO_TRAINING["DPOTrainer\n(Align to Preferences)"]
        
        SFT_MODEL(("SFT Model\nTwinLlama-3.1-8B"))
        DPO_MODEL(("DPO Model\nTwinLlama-3.1-8B-DPO"))
        
        BASE_MODEL -->|Attach LoRA Adapters| SFT_TRAINING
        SFT_DATA -->|Format to Alpaca| SFT_TRAINING
        SFT_TRAINING -->|Save & Merge Weights| SFT_MODEL
        
        SFT_MODEL -->|Attach LoRA Adapters| DPO_TRAINING
        DPO_DATA -->|Format Chosen/Rejected| DPO_TRAINING
        DPO_TRAINING -->|Save & Merge Weights| DPO_MODEL
    end

    subgraph Hugging Face Hub
        HF_SFT[("🤗 saha2026/TwinLlama-3.1-8B")]
        HF_DPO[("🤗 saha2026/TwinLlama-3.1-8B-DPO")]
    end

    SFT_MODEL -->|push_to_hub| HF_SFT
    DPO_MODEL -->|push_to_hub| HF_DPO
```

---

## 3. How QLoRA Works in Our Code

Instead of training all 8 Billion parameters (which would require massive server clusters), we strictly froze the 4-bit base Llama 3.1 model and only trained tiny "Adapter" layers injected into the attention mechanism. 

```mermaid
sequenceDiagram
    participant Script as finetune.py
    participant Unsloth as FastLanguageModel
    participant HF as Hugging Face Hub
    
    Script->>Unsloth: from_pretrained(unsloth/Meta-Llama-3.1-8B, load_in_4bit=True)
    Unsloth->>HF: Download 4-bit Quantized Weights
    HF-->>Unsloth: 8B Parameters (Frozen)
    
    Script->>Unsloth: get_peft_model(r=32, target_modules=['q_proj', 'k_proj', ...])
    Note right of Unsloth: Injects trainable LoRA matrices<br/>(Rank 32) into 7 attention layers.<br/>Only ~0.1% of params are trained.
    Unsloth-->>Script: PeftModel (Ready for Training)
```

**LoRA Target Modules configured:**
- `q_proj` (Query), `k_proj` (Key), `v_proj` (Value), `o_proj` (Output)
- `gate_proj`, `up_proj`, `down_proj` (MLP layers)

---

## 4. The SFT (Supervised Fine-Tuning) Pipeline

During SFT, the model learns **how** to answer questions and **what** persona to adopt.

```mermaid
flowchart LR
    subgraph Raw Data
        INSTRUCTION["Instruction: 'Write a python script...'"]
        OUTPUT["Output: 'Here is the code...'"]
    end

    subgraph Formatter
        ALPACA["Alpaca Template\n\nBelow is an instruction...\n### Instruction: {instruction}\n### Response: {output}<|EOS|>"]
    end

    subgraph Trainer
        PACKING["Sequence Packing\n(Combine multiple short\nprompts up to 2048 tokens)"]
        GRADIENTS["AdamW 8-bit Optimizer\nLR: 3e-4\nBatch: 16 (2x8)"]
    end

    INSTRUCTION & OUTPUT --> ALPACA
    ALPACA -->|Tokenized| PACKING
    PACKING --> GRADIENTS
    GRADIENTS -->|"Log Metrics"| COMET[("☄️ Comet ML")]
```

### Dataset Merging Strategy
We explicitly concatenating two datasets before passing them to the trainer:
1. `saha2026/llmtwin` (Your custom extracted corpus, ~200 specific samples).
2. `mlabonne/FineTome-Alpaca-100k` (A massive open-source dataset, trimmed to 10k samples).
*This prevents "catastrophic forgetting" where the model learns your persona but forgets how to be a general helpful assistant!*

---

## 5. Post-Training: Saving and Inference

Once the `SFTTrainer` finishes its 3 epochs, the script performs three critical final steps:

```mermaid
stateDiagram-v2
    [*] --> Training_Complete
    
    Training_Complete --> Merge_Weights
    note right of Merge_Weights
        Combines the 4-bit base model
        with the trained 16-bit LoRA adapters
        into a single standalone model.
    end note
    
    Merge_Weights --> Push_To_Hub
    note right of Push_To_Hub
        Uploads directly to Hugging Face
        as saha2026/TwinLlama-3.1-8B
    end note
    
    Push_To_Hub --> Local_Inference
    note right of Local_Inference
        Switches Unsloth to Fast Inference mode,
        monkey-patches Transformers cache bugs,
        and generates a test response stream.
    end note
    
    Local_Inference --> [*]
```

### The `_patch_dynamic_cache` Hack
During inference, older versions of Hugging Face Transformers crash when Unsloth tries to use optimized caching. In `finetune.py`, we explicitly append `"dynamic"` to `ALL_CACHE_IMPLEMENTATIONS` right before calling `model.generate()`. This allows the SageMaker instance to successfully complete the pipeline validation locally without crashing at the finish line!
