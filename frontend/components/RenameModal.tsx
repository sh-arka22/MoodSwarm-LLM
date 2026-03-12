import React, { useState } from "react";
import {
  Modal,
  View,
  Text,
  TextInput,
  Pressable,
  StyleSheet,
} from "react-native";
import { colors, spacing, borderRadius, fontSize } from "../constants/theme";

interface Props {
  visible: boolean;
  currentTitle: string;
  onSave: (title: string) => void;
  onCancel: () => void;
}

export function RenameModal({ visible, currentTitle, onSave, onCancel }: Props) {
  const [title, setTitle] = useState(currentTitle);

  return (
    <Modal visible={visible} transparent animationType="fade">
      <View style={styles.overlay}>
        <View style={styles.modal}>
          <Text style={styles.heading}>Rename Conversation</Text>
          <TextInput
            style={styles.input}
            value={title}
            onChangeText={setTitle}
            autoFocus
            selectTextOnFocus
            placeholderTextColor={colors.textMuted}
          />
          <View style={styles.buttons}>
            <Pressable style={styles.cancelBtn} onPress={onCancel}>
              <Text style={styles.cancelText}>Cancel</Text>
            </Pressable>
            <Pressable
              style={styles.saveBtn}
              onPress={() => {
                if (title.trim()) onSave(title.trim());
              }}
            >
              <Text style={styles.saveText}>Save</Text>
            </Pressable>
          </View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.6)",
    justifyContent: "center",
    alignItems: "center",
  },
  modal: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    width: "85%",
    maxWidth: 400,
  },
  heading: {
    color: colors.text,
    fontSize: fontSize.lg,
    fontWeight: "bold",
    marginBottom: spacing.lg,
  },
  input: {
    backgroundColor: colors.background,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    color: colors.text,
    fontSize: fontSize.md,
    marginBottom: spacing.lg,
  },
  buttons: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: spacing.md,
  },
  cancelBtn: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
  },
  cancelText: {
    color: colors.textSecondary,
    fontSize: fontSize.md,
  },
  saveBtn: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    backgroundColor: colors.accent,
    borderRadius: borderRadius.sm,
  },
  saveText: {
    color: "#fff",
    fontSize: fontSize.md,
    fontWeight: "600",
  },
});
