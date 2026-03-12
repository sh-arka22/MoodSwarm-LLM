from datetime import datetime, timezone

from pydantic import UUID4, Field

from .base import NoSQLBaseDocument


class ConversationDocument(NoSQLBaseDocument):
    title: str = "New Chat"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "conversations"


class MessageDocument(NoSQLBaseDocument):
    conversation_id: UUID4
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "messages"
