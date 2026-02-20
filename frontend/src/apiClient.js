/**
 * API client utilities.
 *
 * Requirements:
 * - Route all backend calls through VITE_API_BASE_URL.
 * - Normalize API errors into JavaScript Error objects.
 * - Keep backend URL configuration centralized in one file.
 */
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8080";

console.log("[frontend] API base URL:", API_BASE_URL);

function toApiUrl(path) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL.replace(/\/+$/, "")}${normalizedPath}`;
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    if (isJson && payload && typeof payload === "object") {
      throw new Error(payload.error || payload.message || `Request failed (${response.status})`);
    }

    if (typeof payload === "string" && payload.trim()) {
      throw new Error(payload);
    }

    throw new Error(`Request failed (${response.status})`);
  }

  return payload;
}

export async function apiGet(path) {
  const response = await fetch(toApiUrl(path));
  return parseResponse(response);
}

export async function apiPost(path, body) {
  const init = {
    method: "POST",
    headers: {}
  };

  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  const response = await fetch(toApiUrl(path), init);
  return parseResponse(response);
}

export async function apiUpload(path, file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(toApiUrl(path), {
    method: "POST",
    body: formData
  });

  return parseResponse(response);
}

export { API_BASE_URL };
