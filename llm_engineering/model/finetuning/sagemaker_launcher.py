"""SageMaker launcher for fine-tuning jobs.

Submits finetune.py as a SageMaker training job with Unsloth + QLoRA.
Requires AWS credentials and SageMaker role configured in settings.
"""

from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger

try:
    from sagemaker.huggingface import HuggingFace
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.settings import settings

finetuning_dir = Path(__file__).resolve().parent
finetuning_requirements_path = finetuning_dir / "requirements.txt"


def run_finetuning_on_sagemaker(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "saha2026",
    is_dummy: bool = False,
) -> None:
    assert settings.HUGGINGFACE_ACCESS_TOKEN, "Hugging Face access token is required."
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required."
    assert settings.AWS_REGION, "AWS Region is required."
    assert settings.AWS_ACCESS_KEY, "AWS Access Key is required."
    assert settings.AWS_SECRET_KEY, "AWS Secret Key is required."

    import os

    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_REGION
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_KEY

    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")
    if not finetuning_requirements_path.exists():
        raise FileNotFoundError(f"The file {finetuning_requirements_path} does not exist.")

    api = HfApi()
    user_info = api.whoami(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    huggingface_user = user_info["name"]
    logger.info(f"Current Hugging Face user: {huggingface_user}")

    hyperparameters = {
        "finetuning_type": finetuning_type,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "dataset_huggingface_workspace": dataset_huggingface_workspace,
        "model_output_huggingface_workspace": huggingface_user,
    }
    if is_dummy:
        hyperparameters["is_dummy"] = True

    huggingface_estimator = HuggingFace(
        entry_point="finetune.py",
        source_dir=str(finetuning_dir),
        instance_type=settings.GPU_INSTANCE_TYPE,
        instance_count=1,
        role=settings.AWS_ARN_ROLE,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        hyperparameters=hyperparameters,
        requirements_file=finetuning_requirements_path,
        environment={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY,
            "COMET_PROJECT_NAME": settings.COMET_PROJECT,
        },
    )

    huggingface_estimator.fit()


if __name__ == "__main__":
    run_finetuning_on_sagemaker()
