import React, { useState, useEffect, useRef, useContext } from "react";
import {
  View,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  ActivityIndicator,
  Text,
} from "react-native";
import { useLocalSearchParams, useNavigation } from "expo-router";
import { colors, spacing } from "../../constants/theme";
import { Message } from "../../lib/types";
import * as api from "../../lib/api";
import { ChatBubble } from "../../components/ChatBubble";
import { ChatInput } from "../../components/ChatInput";
import { EmptyState } from "../../components/EmptyState";
import { RefreshContext } from "../_layout";

export default function ChatScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const navigation = useNavigation();
  const flatListRef = useRef<FlatList>(null);
  const { triggerRefresh } = useContext(RefreshContext);

  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(
    id === "new" ? null : id ?? null
  );
  const [error, setError] = useState<string | null>(null);

  // Load messages when conversation changes
  useEffect(() => {
    if (id === "new") {
      setConversationId(null);
      setMessages([]);
      navigation.setOptions({ title: "New Chat" });
      return;
    }
    if (id) {
      setConversationId(id);
      loadMessages(id);
    }
  }, [id]);

  async function loadMessages(convId: string) {
    setLoading(true);
    try {
      const msgs = await api.getMessages(convId);
      setMessages(msgs);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSend(text: string) {
    setError(null);
    setSending(true);

    try {
      let activeConvId = conversationId;

      // Create conversation on first message
      if (!activeConvId) {
        const conv = await api.createConversation("New Chat");
        activeConvId = conv.id;
        setConversationId(activeConvId);
      }

      // Optimistic user message
      const tempUserMsg: Message = {
        id: `temp-${Date.now()}`,
        conversation_id: activeConvId,
        role: "user",
        content: text,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, tempUserMsg]);

      // Send to backend
      const response = await api.sendMessage(activeConvId, text);

      // Replace temp message with real ones
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== tempUserMsg.id),
        response.user_message,
        response.assistant_message,
      ]);

      // Update header title
      const title = text.length > 30 ? text.slice(0, 30) + "..." : text;
      if (messages.length === 0) {
        navigation.setOptions({ title });
      }

      triggerRefresh();
    } catch (e: any) {
      setError(e.message);
      // Remove optimistic message on error
      setMessages((prev) => prev.filter((m) => !m.id.startsWith("temp-")));
    } finally {
      setSending(false);
    }
  }

  function scrollToBottom() {
    if (flatListRef.current && messages.length > 0) {
      flatListRef.current.scrollToEnd({ animated: true });
    }
  }

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  if (loading) {
    return (
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator size="large" color={colors.accent} />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      keyboardVerticalOffset={90}
    >
      {messages.length === 0 && !sending ? (
        <EmptyState onSuggestionPress={handleSend} />
      ) : (
        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => <ChatBubble message={item} />}
          contentContainerStyle={styles.messageList}
          onContentSizeChange={scrollToBottom}
        />
      )}

      {sending && (
        <View style={styles.typingRow}>
          <ActivityIndicator size="small" color={colors.accent} />
          <Text style={styles.typingText}>Thinking...</Text>
        </View>
      )}

      {error && (
        <View style={styles.errorRow}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      <ChatInput onSend={handleSend} disabled={sending} />
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  center: {
    justifyContent: "center",
    alignItems: "center",
  },
  messageList: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
  },
  typingRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    gap: spacing.sm,
  },
  typingText: {
    color: colors.textSecondary,
    fontSize: 13,
  },
  errorRow: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  errorText: {
    color: colors.danger,
    fontSize: 13,
  },
});
