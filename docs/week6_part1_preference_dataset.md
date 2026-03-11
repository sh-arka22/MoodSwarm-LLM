# Week 6 Part 1 — DPO Preference Dataset Generation

**Chapter:** 6 (Preference Alignment) | **Status:** Complete | **Date:** 2026-03-11

## Overview

Week 6 Part 1 generates the preference dataset needed for Direct Preference Optimization (DPO) training. DPO teaches the model to prefer one style of answer (your blog writing) over another (generic LLM output) by training on `(instruction, chosen, rejected)` triples.

| Metric | Value |
|--------|-------|
| Source documents | 7 cleaned articles from Qdrant |
| Raw prompts generated | 142 |
| Raw samples from LLM | 590 |
| Samples after filtering | 80 (86% rejection rate) |
| Train / Test split | 71 / 9 (5% test) |
| Generation cost | ~$0.04 (GPT-4o-mini) |
| Evaluation cost | ~$0.005 |
| HuggingFace dataset | `saha2026/llmtwin-dpo` |

## What DPO Preference Data Looks Like

Each sample is a triple:

```
instruction: "Explain how RAG systems retrieve relevant context."
chosen:      "RAG works by embedding your query and searching a vector store..."
             (verbatim extract from your articles — casual blog tone)
rejected:    "Retrieval-Augmented Generation (RAG) is a methodology that..."
             (LLM-generated answer — formal, generic AI tone)
```

DPO uses these pairs to shift the model's probability distribution: increase likelihood of generating "chosen"-style responses and decrease likelihood of "rejected"-style responses, without needing a separate reward model (unlike RLHF).

## Architecture

### Pipeline DAG

```
query_feature_store          Fetch cleaned docs from Qdrant
        |
        v
  create_prompts             Chunk docs (1000-2000 chars) → LLM prompt templates
        |
        v
generate_preference_dataset  GPT-4o-mini generates 5 triples per extract
        |
        v
  [post-processing]          filter_short_answers → filter_answer_format → train/test split
        |
        v
push_to_huggingface          Upload to saha2026/llmtwin-dpo
```

### Key Files

```
llm_engineering/
├── domain/
│   ├── dataset.py                          Domain models (PreferenceDatasetSample, PreferenceDataset, PreferenceTrainTestSplit)
│   └── prompt.py                           Prompt + GenerateDatasetSamplesPrompt models
├── application/dataset/
│   ├── generation.py                       DatasetGenerator ABC → PreferenceDatasetGenerator
│   ├── utils.py                            Filtering, dedup, contamination, diversity analysis
│   ├── output_parsers.py                   ListPydanticOutputParser (JSON array → Pydantic)
│   └── constants.py                        Mock responses for test mode
├── model/evaluation/
│   └── evaluate.py                         LLM-as-judge (accuracy + style, 1-3 scale)

pipelines/
└── generate_datasets.py                    ZenML pipeline definition

steps/generate_datasets/
├── query_feature_store.py                  Fetch cleaned docs from Qdrant (concurrent per type)
├── create_prompts.py                       Document → prompt generation
├── generate_preference_dataset.py          LLM generation + post-processing
└── push_to_huggingface.py                  HF Hub upload

tools/
├── dataset_inspect.py                      CLI: stats, samples, quality, evaluate, generate
└── push_dataset.py                         Standalone HF push (dry-run supported)

configs/
└── generate_preference_datasets.yaml       test_split=0.05, mock=false
```

## How Each Stage Works

### Stage 1: Feature Store Query

**File:** `steps/generate_datasets/query_feature_store.py`

Fetches all cleaned documents from Qdrant using `ThreadPoolExecutor` to query three collections in parallel:

- `CleanedArticleDocument.bulk_find()`
- `CleanedPostDocument.bulk_find()`
- `CleanedRepositoryDocument.bulk_find()`

Returns a flat list of `CleanedDocument` objects. In our case: 7 cleaned articles.

### Stage 2: Prompt Generation

**Files:** `steps/generate_datasets/create_prompts.py`, `generation.py:get_prompts()`

Each document goes through `extract_substrings()` which splits it into 1000-2000 character chunks. This is separate from the embedding chunking (Week 3) — these are larger extracts designed to give the LLM enough context to generate meaningful Q&A pairs.

For 7 documents, this produced 142 prompt extracts, grouped by `DataCategory` (articles, posts, repositories).

Each prompt uses this template (simplified):

```
Given the following context from a blog post, generate 5 (instruction, rejected, chosen) triples.

- The "chosen" answer must be a VERBATIM copy from the context
- The "rejected" answer should be a generic LLM-style response
- Output as a JSON array

Context: {document_extract}
```

The "verbatim copy" instruction is critical — it ensures "chosen" answers preserve the author's original writing style, which is exactly what DPO will optimize the model toward.

### Stage 3: LLM Generation

**Files:** `steps/generate_datasets/generate_preference_dataset.py`, `generation.py:generate()`

The generation engine:

1. Converts each prompt to LangChain message pairs: `[SystemMessage, HumanMessage]`
2. Batches into groups of 24 for efficient API throughput
3. Calls `ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=2000)`
4. Parses JSON arrays via `ListPydanticOutputParser` → `list[PreferenceDatasetSample]`
5. Flattens all batches into `dict[DataCategory, PreferenceDataset]`

142 prompts x 5 samples each = 590 raw samples (~$0.04).

**Mock mode:** For testing, `FakeListLLM` returns canned responses from `constants.py`, enabling full pipeline validation without API costs.

### Stage 4: Post-Processing & Filtering

**File:** `generation.py:post_process_datasets()`, `utils.py`

Three filters applied in sequence:

| Filter | Logic | Purpose |
|--------|-------|---------|
| `filter_short_answers(min_length=100)` | Remove if `len(chosen) < 100` chars | Eliminate trivially short extracts |
| `filter_answer_format()` | Remove if chosen doesn't start uppercase or end with `.!?` | Enforce grammatical consistency |
| `train_test_split(test_size=0.05)` | scikit-learn split, `random_state=42` | Reproducible evaluation set |

**Result:** 590 raw → 80 after filtering (86% rejection rate). This aggressive filtering is expected — many verbatim extracts are sentence fragments or lack proper formatting.

Final split: 71 train / 9 test.

### Stage 5: HuggingFace Push

**Files:** `steps/generate_datasets/push_to_huggingface.py`, `tools/push_dataset.py`

The `PreferenceTrainTestSplit` converts to HuggingFace `DatasetDict` format:

```python
# Columns for preference dataset:
{
    "train": Dataset(columns=["prompt", "chosen", "rejected"]),
    "test":  Dataset(columns=["prompt", "chosen", "rejected"])
}
```

Pushed to `saha2026/llmtwin-dpo` via `push_to_hub()` with HF access token.

The standalone `tools/push_dataset.py` also supports `--dry-run` mode for previewing before upload.

## Quality Validation

### Automated Checks (`tools/dataset_inspect.py quality --deep`)

| Check | Result |
|-------|--------|
| Empty fields | 0 |
| Short answers (<50 chars) | 0 |
| Exact duplicates | 1 |
| Near-duplicates (Jaccard > 0.7) | 3 pairs |
| Train/test contamination (Jaccard > 0.8) | 0 |
| Bad keywords ("context", "extract", etc.) | 9 (cosmetic, not actionable) |
| Format violations | 0 |

**Diversity stats:** Vocabulary size, type-token ratio, unique bigrams, and top-10 words computed to ensure instruction variety.

### LLM-as-Judge Evaluation (`tools/dataset_inspect.py evaluate`)

**File:** `llm_engineering/model/evaluation/evaluate.py`

GPT-4o-mini scores each "chosen" answer on two dimensions:

| Dimension | Scale | Meaning |
|-----------|-------|---------|
| Accuracy | 1-3 | 1=incorrect, 2=mostly correct, 3=fully accurate |
| Style | 1-3 | 1=formal/generic, 2=natural blog tone, 3=excellent casual voice |

Evaluation runs in parallel (`ThreadPoolExecutor`, 4 threads, batches of 5).

**Results on 20 samples:**

| Metric | Score | Target |
|--------|-------|--------|
| Accuracy (avg) | 2.00 | >= 2.0 |
| Style (avg) | 2.00 | >= 2.0 |

Both metrics meet the minimum threshold. Style=2.0 confirms the "chosen" answers read like blog content rather than AI-generated text.

## Design Decisions

### 1. Verbatim Extraction for "Chosen" Answers

The prompt explicitly instructs: *"Ensure that the extracted answer, the chosen one, is a verbatim copy from the context, including all punctuation and apostrophes."*

This is fundamental to DPO — the model learns to mimic the author's actual writing patterns, not a paraphrased version.

### 2. Aggressive Filtering (86% Rejection)

Most raw samples are filtered because verbatim extracts from articles often:
- Are too short (sentence fragments)
- Don't start with uppercase (mid-paragraph extracts)
- Missing terminal punctuation (incomplete sentences)

This is a feature, not a bug. DPO needs high-quality preference pairs, and training on malformed samples would teach bad habits.

### 3. Small Dataset (80 Samples) is Sufficient for DPO

DPO is much lighter than SFT:
- SFT needs thousands of samples (we used 10,638 including FineTome)
- DPO typically needs 100-1000 preference pairs
- 80 samples from 7 articles is on the low end but viable with beta=0.5 (moderate KL penalty)

The small size is a direct consequence of the source corpus (7 documents). More documents would yield proportionally more samples.

### 4. Template Method Pattern

`DatasetGenerator` (ABC) defines the generation workflow. `PreferenceDatasetGenerator` and `InstructionDatasetGenerator` override:
- `prompt_template_str` — different prompt templates
- `post_process_datasets()` — different filtering logic
- `_get_dataset_sample_type()` — different Pydantic models

This keeps the batching, API calling, and parsing logic shared.

### 5. Deterministic Splits

`random_state=42` in `train_test_split()` ensures identical splits across runs, critical for:
- Reproducible evaluation
- Contamination checking
- Comparing results across experiments

## Commands Reference

```bash
# Generate preference dataset (real mode, ~$0.04)
poetry run python -m tools.run --run-generate-preference-datasets --no-cache

# Or generate via CLI tool (saves to data/preference_dataset_samples.json)
poetry run python -m tools.dataset_inspect generate --type preference

# Inspect statistics
poetry run python -m tools.dataset_inspect stats --type preference

# Browse samples
poetry run python -m tools.dataset_inspect samples --type preference --n 5

# Quality check (basic + deep)
poetry run python -m tools.dataset_inspect quality --type preference --deep

# LLM-as-judge evaluation (~$0.005 for 20 samples)
poetry run python -m tools.dataset_inspect evaluate --type preference --n 20

# Push to HuggingFace (with dry-run option)
poetry run python -m tools.push_dataset \
  --dataset-path data/preference_dataset_samples.json \
  --dataset-id saha2026/llmtwin-dpo \
  --dataset-type preference

# Dry run (preview without uploading)
poetry run python -m tools.push_dataset \
  --dataset-path data/preference_dataset_samples.json \
  --dataset-id saha2026/llmtwin-dpo \
  --dataset-type preference \
  --dry-run
```

## What Comes Next (Week 6 Part 2)

The preference dataset at `saha2026/llmtwin-dpo` feeds directly into DPO training:

1. **DPO Training** on SageMaker with `finetuning_type="dpo"`
2. Key hyperparameters: `beta=0.5`, `lr=2e-6`, 1 epoch
3. Base model: `saha2026/TwinLlama-3.1-8B` (SFT checkpoint from Week 5)
4. Output: `saha2026/TwinLlama-3.1-8B-DPO`
5. Comparative evaluation: SFT vs DPO on accuracy + style scores
