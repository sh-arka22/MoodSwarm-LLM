import React, { useState } from "react";
import {
  View,
  TextInput,
  Pressable,
  Text,
  StyleSheet,
  Platform,
} from "react-native";
import { colors, spacing, borderRadius, fontSize } from "../constants/theme";

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: Props) {
  const [text, setText] = useState("");

  function handleSend() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  }

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Ask anything..."
        placeholderTextColor={colors.textMuted}
        value={text}
        onChangeText={setText}
        multiline
        maxLength={2000}
        editable={!disabled}
        onSubmitEditing={handleSend}
        blurOnSubmit={Platform.OS !== "web"}
      />
      <Pressable
        onPress={handleSend}
        disabled={disabled || !text.trim()}
        style={[
          styles.sendBtn,
          disabled || !text.trim() ? styles.sendBtnDisabled : null,
        ]}
      >
        <Text style={styles.sendIcon}>&#x2191;</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    alignItems: "flex-end",
    padding: spacing.md,
    paddingBottom: spacing.lg,
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    gap: spacing.sm,
  },
  input: {
    flex: 1,
    backgroundColor: colors.inputBg,
    borderRadius: borderRadius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    color: colors.text,
    fontSize: fontSize.md,
    maxHeight: 120,
  },
  sendBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  sendBtnDisabled: {
    backgroundColor: colors.textMuted,
  },
  sendIcon: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "bold",
  },
});
