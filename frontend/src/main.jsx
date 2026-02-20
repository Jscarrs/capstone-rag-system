/**
 * Frontend entry point.
 *
 * Requirements:
 * - Mount React app into the root element.
 * - Load global styles once at startup.
 */
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
