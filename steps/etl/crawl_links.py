from urllib.parse import urlparse

from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher
from llm_engineering.domain.documents import UserDocument


@step
def crawl_links(user: UserDocument, links: list[str]) -> Annotated[list[str], "crawled_links"]:
    dispatcher = CrawlerDispatcher.build().register_medium().register_github()

    logger.info(f"Starting to crawl {len(links)} link(s).")

    metadata = {}
    successful_crawls = 0
    for link in tqdm(links):
        successful_crawl, crawled_domain = _crawl_link(dispatcher, link, user)
        successful_crawls += successful_crawl
        metadata = _add_to_metadata(metadata, crawled_domain, successful_crawl)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="crawled_links", metadata=metadata)

    logger.info(f"Successfully crawled {successful_crawls} / {len(links)} links.")

    return links


def _crawl_link(dispatcher: CrawlerDispatcher, link: str, user: UserDocument) -> tuple[bool, str]:
    crawler = dispatcher.get_crawler(link)
    crawler_domain = urlparse(link).netloc

    try:
        _crawl_with_retry(crawler, link, user)
        return (True, crawler_domain)
    except Exception as e:
        logger.error(f"Failed to crawl {link} after retries: {e!s}")
        return (False, crawler_domain)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Crawl attempt {retry_state.attempt_number} failed, retrying in "
        f"{retry_state.next_action.sleep:.1f}s..."
    ),
    reraise=True,
)
def _crawl_with_retry(crawler, link: str, user) -> None:
    crawler.extract(link=link, user=user)


def _add_to_metadata(metadata: dict, domain: str, successful_crawl: bool) -> dict:
    if domain not in metadata:
        metadata[domain] = {}
    metadata[domain]["successful"] = metadata[domain].get("successful", 0) + successful_crawl
    metadata[domain]["total"] = metadata[domain].get("total", 0) + 1
    return metadata
