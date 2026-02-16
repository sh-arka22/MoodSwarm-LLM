import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from llm_engineering.domain.documents import RepositoryDocument

from .base import BaseCrawler


class GithubCrawler(BaseCrawler):
    model = RepositoryDocument

    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        super().__init__()
        self._ignore = ignore

    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Repository already exists in the database: {link}")
            return

        logger.info(f"Starting scraping GitHub repository: {link}")

        repo_name = link.rstrip("/").split("/")[-1]
        local_temp = tempfile.mkdtemp()

        try:
            subprocess.run(["git", "clone", link], cwd=local_temp, check=True)

            repo_path = Path(local_temp) / os.listdir(local_temp)[0]

            tree = {}
            for root, _, files in os.walk(repo_path):
                dir_rel = str(Path(root).relative_to(repo_path))
                if dir_rel == ".":
                    dir_rel = ""
                if dir_rel.startswith(self._ignore):
                    continue

                for file in files:
                    if file.endswith(self._ignore):
                        continue
                    file_path = str(Path(dir_rel) / file) if dir_rel else file
                    with Path(root, file).open(errors="ignore") as f:
                        tree[file_path] = f.read().replace(" ", "")

            user = kwargs["user"]
            instance = self.model(
                content=tree,
                name=repo_name,
                link=link,
                platform="github",
                author_id=user.id,
                author_full_name=user.full_name,
            )
            instance.save()

        except Exception:
            raise
        finally:
            shutil.rmtree(local_temp)

        logger.info(f"Finished scraping GitHub repository: {link}")
