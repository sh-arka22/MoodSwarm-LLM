from datetime import datetime, timezone

import opik
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from opik import opik_context
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.conversations import ConversationDocument, MessageDocument
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.opik_utils import configure_opik
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint

configure_opik()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Existing RAG models ---


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


# --- Conversation models ---


class CreateConversationRequest(BaseModel):
    title: str = "New Chat"


class RenameConversationRequest(BaseModel):
    title: str


class SendMessageRequest(BaseModel):
    query: str


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str


class ChatResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse


# --- Existing RAG logic ---


@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()

    return answer


@opik.track
def rag(query: str) -> str:
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query, context)

    opik_context.update_current_trace(
        tags=["rag"],
        metadata={
            "model_id": settings.HF_MODEL_ID,
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
            "temperature": settings.TEMPERATURE_INFERENCE,
            "query_tokens": misc.compute_num_tokens(query),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )

    return answer


# --- Existing endpoint (backward compatible) ---


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Helpers ---


def _conv_to_response(conv: ConversationDocument) -> ConversationResponse:
    return ConversationResponse(
        id=str(conv.id),
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
    )


def _msg_to_response(msg: MessageDocument) -> MessageResponse:
    return MessageResponse(
        id=str(msg.id),
        conversation_id=str(msg.conversation_id),
        role=msg.role,
        content=msg.content,
        created_at=msg.created_at.isoformat(),
    )


# --- Conversation endpoints ---


@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: CreateConversationRequest | None = None):
    title = request.title if request else "New Chat"
    conv = ConversationDocument(title=title)
    conv.save()
    return _conv_to_response(conv)


@app.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations():
    convs = ConversationDocument.bulk_find()
    convs.sort(key=lambda c: c.updated_at, reverse=True)
    return [_conv_to_response(c) for c in convs]


@app.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def rename_conversation(conversation_id: str, request: RenameConversationRequest):
    conv = ConversationDocument.find(_id=conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    from llm_engineering.infrastructure.db.mongo import connection
    from llm_engineering.settings import settings as app_settings

    db = connection.get_database(app_settings.DATABASE_NAME)
    db["conversations"].update_one(
        {"_id": conversation_id},
        {"$set": {"title": request.title, "updated_at": datetime.now(timezone.utc).isoformat()}},
    )

    conv = ConversationDocument.find(_id=conversation_id)
    return _conv_to_response(conv)


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conv = ConversationDocument.find(_id=conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    MessageDocument.bulk_delete(conversation_id=conversation_id)
    ConversationDocument.delete(_id=conversation_id)
    return {"detail": "Conversation deleted"}


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages(conversation_id: str):
    conv = ConversationDocument.find(_id=conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = MessageDocument.bulk_find(conversation_id=conversation_id)
    messages.sort(key=lambda m: m.created_at)
    return [_msg_to_response(m) for m in messages]


@app.post("/conversations/{conversation_id}/messages", response_model=ChatResponse)
async def send_message(conversation_id: str, request: SendMessageRequest):
    conv = ConversationDocument.find(_id=conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save user message
    user_msg = MessageDocument(
        conversation_id=conv.id,
        role="user",
        content=request.query,
    )
    user_msg.save()

    # Get RAG response
    try:
        answer = rag(query=request.query)
    except Exception as e:
        answer = f"Sorry, I encountered an error: {e!s}"

    # Save assistant message
    assistant_msg = MessageDocument(
        conversation_id=conv.id,
        role="assistant",
        content=answer,
    )
    assistant_msg.save()

    # Update conversation timestamp and auto-title if still default
    from llm_engineering.infrastructure.db.mongo import connection
    from llm_engineering.settings import settings as app_settings

    db = connection.get_database(app_settings.DATABASE_NAME)
    update_fields: dict = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if conv.title == "New Chat":
        update_fields["title"] = request.query[:50].strip()
    db["conversations"].update_one({"_id": str(conv.id)}, {"$set": update_fields})

    return ChatResponse(
        user_message=_msg_to_response(user_msg),
        assistant_message=_msg_to_response(assistant_msg),
    )
