import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { Message } from "../lib/types";
import { colors, spacing, borderRadius, fontSize } from "../constants/theme";

interface Props {
  message: Message;
}

export function ChatBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <View
      style={[styles.row, isUser ? styles.rowUser : styles.rowAssistant]}
    >
      <View
        style={[
          styles.bubble,
          isUser ? styles.bubbleUser : styles.bubbleAssistant,
        ]}
      >
        <Text style={styles.content}>{message.content}</Text>
        <Text style={styles.timestamp}>
          {new Date(message.created_at).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    marginBottom: spacing.md,
    flexDirection: "row",
  },
  rowUser: {
    justifyContent: "flex-end",
  },
  rowAssistant: {
    justifyContent: "flex-start",
  },
  bubble: {
    maxWidth: "80%",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
  },
  bubbleUser: {
    backgroundColor: colors.userBubble,
    borderBottomRightRadius: borderRadius.sm,
  },
  bubbleAssistant: {
    backgroundColor: colors.assistantBubble,
    borderBottomLeftRadius: borderRadius.sm,
  },
  content: {
    color: colors.text,
    fontSize: fontSize.md,
    lineHeight: 22,
  },
  timestamp: {
    color: colors.textMuted,
    fontSize: fontSize.xs,
    marginTop: spacing.xs,
    alignSelf: "flex-end",
  },
});
