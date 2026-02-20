/**
 * Vite Configuration
 *
 * Requirements:
 * - Serve React frontend on a dedicated development port.
 * - Keep frontend and backend fully separated.
 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173
  }
});
