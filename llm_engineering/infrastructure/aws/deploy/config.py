import json

from loguru import logger

try:
    from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.settings import settings

hugging_face_deploy_config = {
    "HF_MODEL_ID": settings.HF_MODEL_ID,
    "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
    "SM_NUM_GPUS": json.dumps(settings.SM_NUM_GPUS),
    "MAX_INPUT_LENGTH": json.dumps(settings.MAX_INPUT_LENGTH),
    "MAX_TOTAL_TOKENS": json.dumps(settings.MAX_TOTAL_TOKENS),
    "MAX_BATCH_TOTAL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "MAX_BATCH_PREFILL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "HF_MODEL_QUANTIZE": "bitsandbytes",
}


model_resource_config = ResourceRequirements(
    requests={
        "copies": settings.COPIES,
        "num_accelerators": settings.GPUS,
        "num_cpus": settings.CPUS,
        "memory": 5 * 1024,
    },
)
