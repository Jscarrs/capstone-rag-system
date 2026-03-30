/**
 * Vite Configuration
 *
 * Requirements:
 * - Serve React frontend on a dedicated development port.
 * - Keep frontend and backend fully separated.
 * - Copy PDF.js cMaps for non-latin character support (Chinese, etc.)
 */
import path from "node:path";
import { createRequire } from "node:module";
import { defineConfig, normalizePath } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

const require = createRequire(import.meta.url);
const pdfjsDistPath = path.dirname(require.resolve("pdfjs-dist/package.json"));
const cMapsDir = normalizePath(path.join(pdfjsDistPath, "cmaps"));

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [{ src: cMapsDir, dest: "" }],
    }),
  ],
  server: {
    host: "0.0.0.0",
    port: 5173,
  },
});
