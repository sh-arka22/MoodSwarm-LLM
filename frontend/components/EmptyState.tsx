import React from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";
import { colors, spacing, borderRadius, fontSize } from "../constants/theme";

const suggestions = [
  "How do RAG systems work?",
  "Explain vector embeddings",
  "What is fine-tuning?",
];

interface Props {
  onSuggestionPress: (text: string) => void;
}

export function EmptyState({ onSuggestionPress }: Props) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>MoodSwarm</Text>
      <Text style={styles.subtitle}>Ask me anything about LLMs</Text>
      <View style={styles.chips}>
        {suggestions.map((s) => (
          <Pressable
            key={s}
            style={styles.chip}
            onPress={() => onSuggestionPress(s)}
          >
            <Text style={styles.chipText}>{s}</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.xl,
  },
  title: {
    color: colors.text,
    fontSize: fontSize.xxl,
    fontWeight: "bold",
    marginBottom: spacing.sm,
  },
  subtitle: {
    color: colors.textSecondary,
    fontSize: fontSize.md,
    marginBottom: spacing.xxl,
  },
  chips: {
    gap: spacing.md,
    width: "100%",
    maxWidth: 400,
  },
  chip: {
    backgroundColor: colors.surfaceLight,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  chipText: {
    color: colors.text,
    fontSize: fontSize.sm,
    textAlign: "center",
  },
});
