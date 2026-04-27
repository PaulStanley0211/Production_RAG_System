import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";

const DARK_KEY = "rag.darkMode.v1";

function Root() {
  const [isDark, setIsDark] = useState<boolean>(() => {
    try {
      const v = localStorage.getItem(DARK_KEY);
      if (v === "1") return true;
      if (v === "0") return false;
    } catch {
      /* ignore */
    }
    // Default to LIGHT regardless of OS preference (per spec: "light is default")
    return false;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) root.classList.add("dark");
    else root.classList.remove("dark");
    try {
      localStorage.setItem(DARK_KEY, isDark ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [isDark]);

  return <App isDark={isDark} onToggleDark={() => setIsDark((d) => !d)} />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
