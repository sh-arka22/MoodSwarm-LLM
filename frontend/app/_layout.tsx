import React, { useState, createContext } from "react";
import { Pressable, Text, StyleSheet } from "react-native";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { Drawer } from "expo-router/drawer";
import { useRouter } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { colors } from "../constants/theme";
import { ConversationList } from "../components/ConversationList";

export const RefreshContext = createContext<{
  refreshKey: number;
  triggerRefresh: () => void;
}>({ refreshKey: 0, triggerRefresh: () => {} });

export default function Layout() {
  const router = useRouter();
  const [refreshKey, setRefreshKey] = useState(0);
  const triggerRefresh = () => setRefreshKey((k) => k + 1);

  return (
    <RefreshContext.Provider value={{ refreshKey, triggerRefresh }}>
      <GestureHandlerRootView style={{ flex: 1 }}>
        <StatusBar style="light" />
        <Drawer
          drawerContent={(props) => (
            <ConversationList
              {...props}
              refreshKey={refreshKey}
              onRefresh={triggerRefresh}
            />
          )}
          screenOptions={{
            headerStyle: { backgroundColor: colors.surface },
            headerTintColor: colors.text,
            drawerStyle: { backgroundColor: colors.background, width: 300 },
            headerRight: () => (
              <Pressable
                onPress={() => {
                  router.push("/chat/new");
                  triggerRefresh();
                }}
                style={styles.newChatBtn}
              >
                <Text style={styles.plusText}>+</Text>
              </Pressable>
            ),
          }}
        >
          <Drawer.Screen name="index" options={{ title: "MoodSwarm" }} />
          <Drawer.Screen name="chat/[id]" options={{ title: "Chat" }} />
        </Drawer>
      </GestureHandlerRootView>
    </RefreshContext.Provider>
  );
}

const styles = StyleSheet.create({
  newChatBtn: {
    marginRight: 16,
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  plusText: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "bold",
    lineHeight: 22,
  },
});
