"""Push locally generated dataset to HuggingFace Hub.

Usage:
    poetry run python -m tools.push_dataset --dataset-path data/instruct_dataset_samples.json --dataset-id arkaj/llmtwin
    poetry run python -m tools.push_dataset --dataset-path data/instruct_dataset_samples.json --dataset-id arkaj/llmtwin --dry-run
"""

import json
from pathlib import Path

import click
from datasets import Dataset, DatasetDict
from loguru import logger

from llm_engineering.settings import settings


def load_samples(dataset_path: Path) -> list[dict]:
    with dataset_path.open() as f:
        return json.load(f)


def samples_to_hf_dataset(samples: list[dict], dataset_type: str = "instruct") -> DatasetDict:
    train_samples = [s for s in samples if s.get("split", "train") == "train"]
    test_samples = [s for s in samples if s.get("split") == "test"]

    if not test_samples:
        from sklearn.model_selection import train_test_split
        train_samples, test_samples = train_test_split(
            samples, test_size=0.1, random_state=42
        )

    def to_hf(samples_list: list[dict]) -> Dataset:
        if dataset_type == "preference":
            return Dataset.from_dict({
                "prompt": [s["instruction"] for s in samples_list],
                "chosen": [s["chosen"] for s in samples_list],
                "rejected": [s["rejected"] for s in samples_list],
            })
        return Dataset.from_dict({
            "instruction": [s["instruction"] for s in samples_list],
            "output": [s["answer"] for s in samples_list],
        })

    return DatasetDict({
        "train": to_hf(train_samples),
        "test": to_hf(test_samples),
    })


@click.command(help="Push a dataset to HuggingFace Hub.")
@click.option("--dataset-path", required=True, type=click.Path(exists=True), help="Path to JSON dataset file.")
@click.option("--dataset-id", required=True, help="HuggingFace dataset ID (e.g., 'username/llmtwin').")
@click.option("--dataset-type", default="instruct", type=click.Choice(["instruct", "preference"]), help="Dataset type.")
@click.option("--dry-run", is_flag=True, default=False, help="Show stats without pushing.")
def main(dataset_path: str, dataset_id: str, dataset_type: str, dry_run: bool) -> None:
    path = Path(dataset_path)
    samples = load_samples(path)
    logger.info(f"Loaded {len(samples)} samples from {path}")

    hf_dataset = samples_to_hf_dataset(samples, dataset_type=dataset_type)

    logger.info(f"Train: {len(hf_dataset['train'])} samples")
    logger.info(f"Test:  {len(hf_dataset['test'])} samples")
    logger.info(f"Columns: {hf_dataset['train'].column_names}")

    logger.info("Sample (train[0]):")
    sample = hf_dataset["train"][0]
    if dataset_type == "preference":
        logger.info(f"  prompt: {sample['prompt'][:80]}...")
        logger.info(f"  chosen: {sample['chosen'][:80]}...")
        logger.info(f"  rejected: {sample['rejected'][:80]}...")
    else:
        logger.info(f"  instruction: {sample['instruction'][:80]}...")
        logger.info(f"  output: {sample['output'][:80]}...")

    if dry_run:
        logger.info("[DRY RUN] Would push to: {}", dataset_id)
        return

    assert settings.HUGGINGFACE_ACCESS_TOKEN, "HUGGINGFACE_ACCESS_TOKEN not set in .env"

    logger.info(f"Pushing to HuggingFace: {dataset_id}")
    hf_dataset.push_to_hub(dataset_id, token=settings.HUGGINGFACE_ACCESS_TOKEN)
    logger.success(f"Dataset pushed to https://huggingface.co/datasets/{dataset_id}")


if __name__ == "__main__":
    main()
