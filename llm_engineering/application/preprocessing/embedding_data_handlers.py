from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

from llm_engineering.application.networks import EmbeddingModelSingleton
from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.chunks import ArticleChunk, PostChunk, RepositoryChunk
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)
from llm_engineering.domain.queries import EmbeddedQuery, Query

ChunkT = TypeVar("ChunkT", bound=VectorBaseDocument)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=VectorBaseDocument)

embedding_model = EmbeddingModelSingleton()


class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        return self.embed_batch([data_model])[0]

    def embed_batch(self, data_model: list[ChunkT]) -> list[EmbeddedChunkT]:
        embedding_model_input = [dm.content for dm in data_model]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        embedded_chunk = [
            self.map_model(dm, cast(list[float], embedding))
            for dm, embedding in zip(data_model, embeddings, strict=False)
        ]

        return embedded_chunk

    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass


class PostEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: PostChunk, embedding: list[float]) -> EmbeddedPostChunk:
        return EmbeddedPostChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class ArticleEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: ArticleChunk, embedding: list[float]) -> EmbeddedArticleChunk:
        return EmbeddedArticleChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class RepositoryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: RepositoryChunk, embedding: list[float]) -> EmbeddedRepositoryChunk:
        return EmbeddedRepositoryChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            name=data_model.name,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class QueryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: Query, embedding: list[float]) -> EmbeddedQuery:
        return EmbeddedQuery(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )
