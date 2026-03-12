import React, { useState } from "react";
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  Alert,
  Platform,
} from "react-native";
import { Conversation } from "../lib/types";
import { colors, spacing, borderRadius, fontSize } from "../constants/theme";

interface Props {
  conversation: Conversation;
  active: boolean;
  onPress: () => void;
  onRename: () => void;
  onDelete: () => void;
}

export function ConversationItem({
  conversation,
  active,
  onPress,
  onRename,
  onDelete,
}: Props) {
  const [showActions, setShowActions] = useState(false);

  function handleLongPress() {
    if (Platform.OS === "web") {
      setShowActions((v) => !v);
    } else {
      Alert.alert(conversation.title, "Choose an action", [
        { text: "Rename", onPress: onRename },
        { text: "Delete", style: "destructive", onPress: onDelete },
        { text: "Cancel", style: "cancel" },
      ]);
    }
  }

  const timeAgo = getRelativeTime(conversation.updated_at);

  return (
    <View>
      <Pressable
        style={[styles.container, active && styles.active]}
        onPress={onPress}
        onLongPress={handleLongPress}
      >
        <Text style={styles.title} numberOfLines={1}>
          {conversation.title}
        </Text>
        <Text style={styles.time}>{timeAgo}</Text>
      </Pressable>

      {showActions && Platform.OS === "web" && (
        <View style={styles.actions}>
          <Pressable style={styles.actionBtn} onPress={onRename}>
            <Text style={styles.actionText}>Rename</Text>
          </Pressable>
          <Pressable
            style={styles.actionBtn}
            onPress={() => {
              setShowActions(false);
              onDelete();
            }}
          >
            <Text style={[styles.actionText, { color: colors.danger }]}>
              Delete
            </Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}

function getRelativeTime(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  return `${days}d`;
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    marginVertical: 2,
    borderRadius: borderRadius.md,
  },
  active: {
    backgroundColor: colors.surfaceLight,
  },
  title: {
    color: colors.text,
    fontSize: fontSize.sm,
    flex: 1,
    marginRight: spacing.sm,
  },
  time: {
    color: colors.textMuted,
    fontSize: fontSize.xs,
  },
  actions: {
    flexDirection: "row",
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.sm,
    gap: spacing.md,
  },
  actionBtn: {
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.sm,
  },
  actionText: {
    color: colors.text,
    fontSize: fontSize.xs,
  },
});
