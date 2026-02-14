from zenml import pipeline, step

from llm_engineering import settings


@step
def check_mongodb_connection() -> str:
    """Connect to MongoDB and list databases."""
    from loguru import logger
    from pymongo import MongoClient

    logger.info(f"Connecting to MongoDB at: {settings.DATABASE_HOST}")
    client = MongoClient(settings.DATABASE_HOST, serverSelectionTimeoutMS=5000)
    db_names = client.list_database_names()
    logger.info(f"MongoDB connected. Databases: {db_names}")
    client.close()

    return f"MongoDB OK — databases: {db_names}"


@step
def check_qdrant_connection() -> str:
    """Connect to Qdrant and list collections."""
    from loguru import logger
    from qdrant_client import QdrantClient

    logger.info(f"Connecting to Qdrant at: {settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}")
    client = QdrantClient(
        host=settings.QDRANT_DATABASE_HOST,
        port=settings.QDRANT_DATABASE_PORT,
    )
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    logger.info(f"Qdrant connected. Collections: {collection_names}")

    return f"Qdrant OK — collections: {collection_names}"


@step
def print_results(mongo_result: str, qdrant_result: str) -> None:
    """Print smoke test results."""
    from loguru import logger

    logger.info("=== SMOKE TEST RESULTS ===")
    logger.info(mongo_result)
    logger.info(qdrant_result)
    logger.info("=== ALL CHECKS PASSED ===")


@pipeline(name="smoke_test")
def smoke_test_pipeline():
    mongo_result = check_mongodb_connection()
    qdrant_result = check_qdrant_connection()
    print_results(mongo_result, qdrant_result)
