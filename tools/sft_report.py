"""SFT training readiness report — validates dataset, configs, and credentials.

Usage:
    poetry run python -m tools.sft_report
    poetry run python -m tools.sft_report --check-hf
    poetry run python -m tools.sft_report --check-aws
"""

import json
from pathlib import Path

import click
from loguru import logger

from llm_engineering.settings import settings

ROOT_DIR = Path(__file__).resolve().parent.parent


def check_dataset(dataset_path: Path) -> dict:
    if not dataset_path.exists():
        return {"status": "MISSING", "path": str(dataset_path)}

    with dataset_path.open() as f:
        samples = json.load(f)

    train = [s for s in samples if s.get("split", "train") == "train"]
    test = [s for s in samples if s.get("split") == "test"]

    return {
        "status": "OK",
        "path": str(dataset_path),
        "total": len(samples),
        "train": len(train),
        "test": len(test),
        "avg_instruction_len": sum(len(s["instruction"]) for s in samples) / max(len(samples), 1),
        "avg_answer_len": sum(len(s["answer"]) for s in samples) / max(len(samples), 1),
    }


def check_configs() -> dict:
    configs = {}
    for name in ["training.yaml", "evaluating.yaml"]:
        path = ROOT_DIR / "configs" / name
        configs[name] = "OK" if path.exists() else "MISSING"
    return configs


def check_files() -> dict:
    files = {}
    paths = [
        "llm_engineering/model/finetuning/finetune.py",
        "llm_engineering/model/finetuning/sagemaker.py",
        "llm_engineering/model/finetuning/requirements.txt",
        "llm_engineering/model/evaluation/evaluate.py",
        "llm_engineering/model/evaluation/sagemaker.py",
        "llm_engineering/model/evaluation/requirements.txt",
        "pipelines/training.py",
        "pipelines/evaluating.py",
        "steps/training/train.py",
        "steps/evaluating/evaluate.py",
    ]
    for p in paths:
        full = ROOT_DIR / p
        files[p] = "OK" if full.exists() else "MISSING"
    return files


def check_credentials(check_hf: bool = False, check_aws: bool = False) -> dict:
    creds = {}
    creds["OPENAI_API_KEY"] = "SET" if settings.OPENAI_API_KEY else "MISSING"
    creds["HUGGINGFACE_ACCESS_TOKEN"] = "SET" if settings.HUGGINGFACE_ACCESS_TOKEN else "MISSING"
    creds["COMET_API_KEY"] = "SET" if settings.COMET_API_KEY else "MISSING (optional)"
    creds["AWS_ARN_ROLE"] = "SET" if settings.AWS_ARN_ROLE else "MISSING (needed for SageMaker)"

    if check_hf and settings.HUGGINGFACE_ACCESS_TOKEN:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            user = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
            creds["HF_USER"] = user["name"]
            creds["HF_AUTH"] = "VERIFIED"
        except Exception as e:
            creds["HF_AUTH"] = f"FAILED: {e}"

    if check_aws and settings.AWS_ARN_ROLE:
        try:
            import boto3

            sts = boto3.client("sts")
            identity = sts.get_caller_identity()
            creds["AWS_ACCOUNT"] = identity["Account"]
            creds["AWS_AUTH"] = "VERIFIED"
        except Exception as e:
            creds["AWS_AUTH"] = f"FAILED: {e}"

    return creds


@click.command(help="SFT training readiness report.")
@click.option("--check-hf", is_flag=True, default=False, help="Verify HuggingFace auth.")
@click.option("--check-aws", is_flag=True, default=False, help="Verify AWS auth.")
def main(check_hf: bool, check_aws: bool) -> None:
    logger.info("=== SFT Training Readiness Report ===\n")

    # 1. Dataset
    dataset_path = ROOT_DIR / "data" / "instruct_dataset_samples.json"
    ds = check_dataset(dataset_path)
    logger.info(f"[Dataset] Status: {ds['status']}")
    if ds["status"] == "OK":
        logger.info(f"  Total: {ds['total']} | Train: {ds['train']} | Test: {ds['test']}")
        logger.info(
            f"  Avg instruction: {ds['avg_instruction_len']:.0f} chars | Avg answer: {ds['avg_answer_len']:.0f} chars"
        )
    logger.info("")

    # 2. Configs
    configs = check_configs()
    logger.info("[Configs]")
    for name, status in configs.items():
        logger.info(f"  {name}: {status}")
    logger.info("")

    # 3. Files
    files = check_files()
    logger.info("[Files]")
    missing = []
    for path, status in files.items():
        if status != "OK":
            missing.append(path)
        logger.info(f"  {path}: {status}")
    logger.info("")

    # 4. Credentials
    creds = check_credentials(check_hf, check_aws)
    logger.info("[Credentials]")
    for key, status in creds.items():
        logger.info(f"  {key}: {status}")
    logger.info("")

    # 5. Training config summary
    training_config = ROOT_DIR / "configs" / "training.yaml"
    if training_config.exists():
        import yaml

        with training_config.open() as f:
            config = yaml.safe_load(f)
        params = config.get("parameters", {})
        logger.info("[Training Config]")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")
        logger.info("")

    # 6. Summary
    all_ok = ds["status"] == "OK" and not missing
    if all_ok:
        logger.success("All checks passed. Ready for training.")
        logger.info("")
        logger.info("Next steps:")
        logger.info(
            "  1. Push dataset:  poetry run python -m tools.push_dataset --dataset-path data/instruct_dataset_samples.json --dataset-id <user>/llmtwin --dry-run"
        )
        logger.info("  2. Run training:  poetry run python -m tools.run --run-training --no-cache")
        logger.info("  3. Run eval:      poetry run python -m tools.run --run-evaluation --no-cache")
    else:
        logger.warning("Some checks failed. Fix issues above before training.")


if __name__ == "__main__":
    main()
