from loguru import logger

try:
    import boto3
    from sagemaker.enums import EndpointType
    from sagemaker.huggingface import get_huggingface_llm_image_uri
    from sagemaker.session import Session
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.model.utils import ResourceManager
from llm_engineering.settings import settings

from .config import hugging_face_deploy_config, model_resource_config
from .sagemaker_huggingface import DeploymentService, SagemakerHuggingfaceStrategy


def create_endpoint(endpoint_type=EndpointType.MODEL_BASED) -> None:
    assert settings.AWS_ARN_ROLE is not None, "AWS_ARN_ROLE is not set in the .env file."

    logger.info(f"Creating endpoint with endpoint_type = {endpoint_type} and model_id = {settings.HF_MODEL_ID}")

    boto_session = boto3.Session(
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )
    sagemaker_session = Session(boto_session=boto_session)
    llm_image = get_huggingface_llm_image_uri("huggingface", version="2.4.0", session=sagemaker_session)

    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager=resource_manager)

    SagemakerHuggingfaceStrategy(deployment_service).deploy(
        role_arn=settings.AWS_ARN_ROLE,
        llm_image=llm_image,
        config=hugging_face_deploy_config,
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        endpoint_config_name=settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE,
        gpu_instance_type=settings.GPU_INSTANCE_TYPE,
        resources=model_resource_config,
        endpoint_type=endpoint_type,
        sagemaker_session=sagemaker_session,
    )
