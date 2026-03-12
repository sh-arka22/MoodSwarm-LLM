import enum
from typing import Optional

from loguru import logger

try:
    import boto3
    from sagemaker.enums import EndpointType
    from sagemaker.huggingface import HuggingFaceModel
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS or SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.domain.inference import DeploymentStrategy
from llm_engineering.settings import settings


class SagemakerHuggingfaceStrategy(DeploymentStrategy):
    def __init__(self, deployment_service) -> None:
        self.deployment_service = deployment_service

    def deploy(
        self,
        role_arn: str,
        llm_image: str,
        config: dict,
        endpoint_name: str,
        endpoint_config_name: str,
        gpu_instance_type: str,
        resources: Optional[dict] = None,
        endpoint_type: enum.Enum = EndpointType.MODEL_BASED,
    ) -> None:
        logger.info("Starting deployment using Sagemaker Huggingface Strategy...")
        logger.info(
            f"Deployment parameters: nb of replicas: {settings.COPIES}, nb of gpus:{settings.GPUS}, "
            f"instance_type:{settings.GPU_INSTANCE_TYPE}"
        )
        try:
            self.deployment_service.deploy(
                role_arn=role_arn,
                llm_image=llm_image,
                config=config,
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_config_name,
                gpu_instance_type=gpu_instance_type,
                resources=resources,
                endpoint_type=endpoint_type,
            )
            logger.info("Deployment completed successfully.")
        except Exception as e:
            logger.error(f"Error during deployment: {e}")
            raise


class DeploymentService:
    def __init__(self, resource_manager):
        self.sagemaker_client = boto3.client(
            "sagemaker",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
        )
        self.resource_manager = resource_manager

    def deploy(
        self,
        role_arn: str,
        llm_image: str,
        config: dict,
        endpoint_name: str,
        endpoint_config_name: str,
        gpu_instance_type: str,
        resources: Optional[dict] = None,
        endpoint_type: enum.Enum = EndpointType.MODEL_BASED,
    ) -> None:
        try:
            if self.resource_manager.endpoint_config_exists(endpoint_config_name=endpoint_config_name):
                logger.info(f"Endpoint configuration {endpoint_config_name} exists. Using existing configuration...")
            else:
                logger.info(f"Endpoint configuration {endpoint_config_name} does not exist.")

            self.prepare_and_deploy_model(
                role_arn=role_arn,
                llm_image=llm_image,
                config=config,
                endpoint_name=endpoint_name,
                update_endpoint=False,
                resources=resources,
                endpoint_type=endpoint_type,
                gpu_instance_type=gpu_instance_type,
            )

            logger.info(f"Successfully deployed/updated model to endpoint {endpoint_name}.")
        except Exception as e:
            logger.error(f"Failed to deploy model to SageMaker: {e}")

            raise

    @staticmethod
    def prepare_and_deploy_model(
        role_arn: str,
        llm_image: str,
        config: dict,
        endpoint_name: str,
        update_endpoint: bool,
        gpu_instance_type: str,
        resources: Optional[dict] = None,
        endpoint_type: enum.Enum = EndpointType.MODEL_BASED,
    ) -> None:
        huggingface_model = HuggingFaceModel(
            role=role_arn,
            image_uri=llm_image,
            env=config,
        )

        huggingface_model.deploy(
            instance_type=gpu_instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            update_endpoint=update_endpoint,
            resources=resources,
            tags=[{"Key": "task", "Value": "model_task"}],
            endpoint_type=endpoint_type,
            container_startup_health_check_timeout=900,
        )
