import { Conversation, Message, ChatResponse } from "./types";

const BASE_URL = "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

export async function createConversation(
  title = "New Chat"
): Promise<Conversation> {
  return request("/conversations", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
}

export async function listConversations(): Promise<Conversation[]> {
  return request("/conversations");
}

export async function renameConversation(
  id: string,
  title: string
): Promise<Conversation> {
  return request(`/conversations/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function deleteConversation(id: string): Promise<void> {
  await request(`/conversations/${id}`, { method: "DELETE" });
}

export async function getMessages(conversationId: string): Promise<Message[]> {
  return request(`/conversations/${conversationId}/messages`);
}

export async function sendMessage(
  conversationId: string,
  query: string
): Promise<ChatResponse> {
  return request(`/conversations/${conversationId}/messages`, {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}
