import { API_BASE_URL } from "./config.js";

// Deliberately in-memory only, not localStorage: the backend session
// (documents + chat memory) survives for as long as this id is reused, but
// the chat log on screen is only ever what's rendered in the current page
// load. Persisting the id across reloads made those two fall out of sync —
// a refresh looked like a fresh start but was still talking to the old,
// invisible session underneath, so an answer could be shaped by history
// the user could no longer see. Regenerating on every load keeps what's
// visible and what the backend remembers consistent.
let sessionId = null;

export function getSessionId() {
  if (!sessionId) sessionId = crypto.randomUUID();
  return sessionId;
}

export function resetSessionId() {
  sessionId = crypto.randomUUID();
  return sessionId;
}

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "X-Session-Id": getSessionId(),
      ...options.headers,
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON; fall back to statusText
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function processDocuments(files) {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  return request("/api/documents", { method: "POST", body: form });
}

export async function resetDocuments() {
  return request("/api/documents", { method: "DELETE" });
}

export async function sendChatMessage(question) {
  return request("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

/**
 * Polls /api/health until the backend responds (Render free-tier instances
 * sleep after ~15 min idle and can take up to ~50s to wake on the next
 * request). Calls onAttempt(attemptNumber) before each try so the UI can
 * show a "waking up" message once it's taking a while.
 */
export async function waitForBackend({ onAttempt, timeoutMs = 90_000, intervalMs = 3_000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let attempt = 0;
  while (Date.now() < deadline) {
    attempt += 1;
    onAttempt?.(attempt);
    try {
      const res = await fetch(`${API_BASE_URL}/api/health`, { signal: AbortSignal.timeout(intervalMs) });
      if (res.ok) return true;
    } catch {
      // still waking up / unreachable — retry until the deadline
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return false;
}
