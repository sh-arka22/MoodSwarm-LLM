"""CLI for managing SageMaker inference endpoints.

Usage:
    poetry run python -m tools.deploy_endpoint create    # Deploy endpoint
    poetry run python -m tools.deploy_endpoint delete    # Teardown endpoint
    poetry run python -m tools.deploy_endpoint status    # Check endpoint status
"""

import sys

from loguru import logger

try:
    import boto3
    from botocore.exceptions import ClientError
except ModuleNotFoundError:
    logger.error("AWS dependencies not installed. Run 'poetry install --with aws'.")
    sys.exit(1)

from llm_engineering.settings import settings


def create():
    """Deploy the inference endpoint to SageMaker."""
    from llm_engineering.infrastructure.aws.deploy.run import create_endpoint

    logger.info(f"Deploying model {settings.HF_MODEL_ID} to endpoint '{settings.SAGEMAKER_ENDPOINT_INFERENCE}'...")
    logger.warning(
        f"This will create a {settings.GPU_INSTANCE_TYPE} instance (~$1.20/hr). Remember to delete when done!"
    )
    create_endpoint()
    logger.info("Endpoint deployment initiated. It may take 5-15 minutes to become InService.")


def delete():
    """Delete the inference endpoint and associated resources."""
    from llm_engineering.infrastructure.aws.deploy.delete_endpoint import delete_endpoint_and_config

    endpoint_name = settings.SAGEMAKER_ENDPOINT_INFERENCE
    logger.info(f"Deleting endpoint '{endpoint_name}' and associated resources...")
    delete_endpoint_and_config(endpoint_name)
    logger.info("Endpoint deletion complete.")


def status():
    """Check the status of the inference endpoint."""
    sagemaker_client = boto3.client(
        "sagemaker",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )
    endpoint_name = settings.SAGEMAKER_ENDPOINT_INFERENCE

    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        logger.info(f"Endpoint '{endpoint_name}': {status}")
        if status == "InService":
            logger.info("Endpoint is ready for inference.")
        elif status == "Creating":
            logger.info("Endpoint is still being created. Please wait...")
        elif status == "Failed":
            reason = response.get("FailureReason", "Unknown")
            logger.error(f"Endpoint creation failed: {reason}")
    except ClientError:
        logger.info(f"Endpoint '{endpoint_name}' does not exist.")


COMMANDS = {
    "create": create,
    "delete": delete,
    "status": status,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        logger.error(f"Usage: python -m tools.deploy_endpoint [{'/'.join(COMMANDS.keys())}]")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
