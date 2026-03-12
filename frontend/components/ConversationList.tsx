import React, { useState, useEffect, useCallback } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  StyleSheet,
  RefreshControl,
} from "react-native";
import { useRouter, usePathname } from "expo-router";
import { DrawerContentComponentProps } from "@react-navigation/drawer";
import { colors, spacing, borderRadius, fontSize } from "../constants/theme";
import { Conversation } from "../lib/types";
import * as api from "../lib/api";
import { ConversationItem } from "./ConversationItem";
import { RenameModal } from "./RenameModal";

interface Props extends DrawerContentComponentProps {
  refreshKey: number;
  onRefresh: () => void;
}

export function ConversationList({ refreshKey, onRefresh, navigation }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [renameTarget, setRenameTarget] = useState<Conversation | null>(null);

  const activeId = pathname.startsWith("/chat/")
    ? pathname.replace("/chat/", "")
    : null;

  const loadConversations = useCallback(async () => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch {
      // silently fail — user will see empty list
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [refreshKey, loadConversations]);

  async function handleRefresh() {
    setRefreshing(true);
    await loadConversations();
    setRefreshing(false);
  }

  async function handleDelete(id: string) {
    await api.deleteConversation(id);
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeId === id) {
      router.push("/chat/new");
    }
    onRefresh();
  }

  async function handleRename(id: string, title: string) {
    await api.renameConversation(id, title);
    setRenameTarget(null);
    await loadConversations();
  }

  return (
    <View style={styles.container}>
      <Pressable
        style={styles.newChatBtn}
        onPress={() => {
          router.push("/chat/new");
          navigation.closeDrawer();
        }}
      >
        <Text style={styles.newChatText}>+ New Chat</Text>
      </Pressable>

      <FlatList
        data={conversations}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <ConversationItem
            conversation={item}
            active={item.id === activeId}
            onPress={() => {
              router.push(`/chat/${item.id}`);
              navigation.closeDrawer();
            }}
            onRename={() => setRenameTarget(item)}
            onDelete={() => handleDelete(item.id)}
          />
        )}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            tintColor={colors.accent}
          />
        }
        contentContainerStyle={styles.list}
      />

      {renameTarget && (
        <RenameModal
          visible={true}
          currentTitle={renameTarget.title}
          onSave={(title) => handleRename(renameTarget.id, title)}
          onCancel={() => setRenameTarget(null)}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    paddingTop: spacing.xxl + spacing.xl,
  },
  newChatBtn: {
    marginHorizontal: spacing.lg,
    marginBottom: spacing.md,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.accent,
    alignItems: "center",
  },
  newChatText: {
    color: colors.accent,
    fontSize: fontSize.md,
    fontWeight: "600",
  },
  list: {
    paddingHorizontal: spacing.sm,
  },
});
