"""Dataset generation utilities — train/test split, filtering, extract chunking, quality checks."""

from collections import Counter
from itertools import pairwise

from sklearn.model_selection import train_test_split

from llm_engineering.application.preprocessing.operations.chunking import chunk_document
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.dataset import (
    InstructDataset,
    InstructDatasetSample,
    InstructTrainTestSplit,
    PreferenceDataset,
    PreferenceDatasetSample,
    PreferenceTrainTestSplit,
)
from llm_engineering.domain.types import DataCategory


def create_instruct_train_test_split(
    data: dict[DataCategory, InstructDataset], test_size=0.2, random_state=42
) -> InstructTrainTestSplit:
    train_data = {}
    test_data = {}

    for category, dataset in data.items():
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        if len(samples_dicts) > 0:
            train_samples_dicts, test_samples_dicts = train_test_split(
                samples_dicts, test_size=test_size, random_state=random_state
            )
            train_samples = [InstructDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
            test_samples = [InstructDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]
        else:
            train_samples = []
            test_samples = []

        train_dataset = InstructDataset(category=category, samples=train_samples)
        test_dataset = InstructDataset(category=category, samples=test_samples)

        train_data[category] = train_dataset
        test_data[category] = test_dataset

    return InstructTrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)


def create_preference_train_test_split(
    data: dict[DataCategory, PreferenceDataset], test_size=0.2, random_state=42
) -> PreferenceTrainTestSplit:
    train_data = {}
    test_data = {}

    for category, dataset in data.items():
        samples = dataset.samples
        samples_dicts = [sample.model_dump() for sample in samples]

        if len(samples_dicts) > 0:
            train_samples_dicts, test_samples_dicts = train_test_split(
                samples_dicts, test_size=test_size, random_state=random_state
            )
            train_samples = [PreferenceDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
            test_samples = [PreferenceDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]
        else:
            train_samples = []
            test_samples = []

        train_dataset = PreferenceDataset(category=category, samples=train_samples)
        test_dataset = PreferenceDataset(category=category, samples=test_samples)

        train_data[category] = train_dataset
        test_data[category] = test_dataset

    return PreferenceTrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)


def filter_short_answers(
    data: dict[DataCategory, PreferenceDataset], min_length: int = 100
) -> dict[DataCategory, PreferenceDataset]:
    def is_long_enough(example: PreferenceDatasetSample) -> bool:
        return len(example.chosen) >= min_length

    filtered_data = {}
    for category, dataset in data.items():
        filtered_dataset_samples = list(filter(is_long_enough, dataset.samples))
        filtered_dataset = PreferenceDataset(category=category, samples=filtered_dataset_samples)
        filtered_data[category] = filtered_dataset

    return filtered_data


def filter_answer_format(data: dict[DataCategory, PreferenceDataset]) -> dict[DataCategory, PreferenceDataset]:
    def is_valid_format(example: PreferenceDatasetSample) -> bool:
        chosen = example.chosen
        return len(chosen) > 0 and chosen[0].isupper() and chosen[-1] in (".", "!", "?")

    filtered_data = {}
    for category, dataset in data.items():
        filtered_dataset_samples = list(filter(is_valid_format, dataset.samples))
        filtered_dataset = PreferenceDataset(category=category, samples=filtered_dataset_samples)
        filtered_data[category] = filtered_dataset

    return filtered_data


def compute_ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute Jaccard similarity of character n-grams between two texts."""
    a_lower = text_a.lower().strip()
    b_lower = text_b.lower().strip()
    if len(a_lower) < n or len(b_lower) < n:
        return 0.0

    ngrams_a = set(a_lower[i : i + n] for i in range(len(a_lower) - n + 1))
    ngrams_b = set(b_lower[i : i + n] for i in range(len(b_lower) - n + 1))

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union) if union else 0.0


def find_near_duplicates(
    samples: list[dict], key: str = "instruction", threshold: float = 0.7, n: int = 3
) -> list[tuple[int, int, float]]:
    """Find near-duplicate pairs using character n-gram Jaccard similarity.

    Returns list of (idx_a, idx_b, similarity) tuples above threshold.
    """
    duplicates = []
    texts = [s[key].strip() for s in samples]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = compute_ngram_overlap(texts[i], texts[j], n=n)
            if sim >= threshold:
                duplicates.append((i, j, sim))
    return duplicates


def check_train_test_contamination(
    samples: list[dict], key: str = "instruction", threshold: float = 0.8, n: int = 3
) -> list[tuple[int, int, float]]:
    """Check if any test instructions are near-duplicates of train instructions.

    Returns list of (train_idx, test_idx, similarity) tuples above threshold.
    """
    train_samples = [(i, s) for i, s in enumerate(samples) if s["split"] == "train"]
    test_samples = [(i, s) for i, s in enumerate(samples) if s["split"] == "test"]

    contaminated = []
    for ti, ts in test_samples:
        for tri, trs in train_samples:
            sim = compute_ngram_overlap(ts[key], trs[key], n=n)
            if sim >= threshold:
                contaminated.append((tri, ti, sim))
    return contaminated


def compute_diversity_stats(samples: list[dict], key: str = "instruction") -> dict:
    """Compute vocabulary diversity and n-gram statistics."""
    all_words = []
    for s in samples:
        all_words.extend(s[key].lower().split())

    word_freq = Counter(all_words)
    vocab_size = len(word_freq)
    total_words = len(all_words)
    type_token_ratio = vocab_size / total_words if total_words > 0 else 0.0

    # Bigram diversity
    bigrams = []
    for s in samples:
        words = s[key].lower().split()
        bigrams.extend(pairwise(words))
    unique_bigrams = len(set(bigrams))

    return {
        "vocab_size": vocab_size,
        "total_words": total_words,
        "type_token_ratio": round(type_token_ratio, 3),
        "unique_bigrams": unique_bigrams,
        "top_10_words": word_freq.most_common(10),
    }


def extract_substrings(
    documents: list[CleanedDocument], min_length: int = 1000, max_length: int = 2000
) -> list[CleanedDocument]:
    extracts = []
    for document in documents:
        document_extracts = chunk_document(document.content, min_length, max_length)
        for extract in document_extracts:
            subdocument = document.model_copy()
            subdocument.content = extract

            extracts.append(subdocument)

    return extracts
