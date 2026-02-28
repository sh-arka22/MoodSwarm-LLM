import opik
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering.application import utils
from llm_engineering.domain.documents import UserDocument
from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .base import RAGStep
from .prompt_templates import SelfQueryTemplate


class SelfQuery(RAGStep):
    @opik.track(name="SelfQuery.generate")
    def generate(self, query: Query) -> Query:
        if self._mock:
            return query

        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )

        chain = prompt | model

        response = chain.invoke({"question": query.content})
        user_full_name = response.content.strip("\n ")

        if user_full_name == "none":
            return query

        first_name, last_name = utils.split_user_full_name(user_full_name)
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

        query.author_id = user.id
        query.author_full_name = user.full_name

        logger.info(f"SelfQuery extracted author: {query.author_full_name} (id={query.author_id})")

        return query
