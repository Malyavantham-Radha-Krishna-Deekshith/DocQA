import { processDocuments, resetDocuments, sendChatMessage, waitForBackend, ApiError } from "./api.js";

const MAX_TRAY_ATTACHMENTS = 20;

const state = {
  attachments: [], // { id, file, url, status: 'processing'|'done'|'error', errorMessage, processingPromise }
};

const el = {
  wakeOverlay: document.getElementById("wake-overlay"),
  wakeMessage: document.getElementById("wake-message"),
  app: document.getElementById("app"),
  toastContainer: document.getElementById("toast-container"),

  newChatBtn: document.getElementById("new-chat-btn"),

  chatLogWrap: document.getElementById("chat-log-wrap"),
  chatLog: document.getElementById("chat-log"),
  chatEmptyState: document.getElementById("chat-empty-state"),

  attachmentTray: document.getElementById("attachment-tray"),
  chatForm: document.getElementById("chat-form"),
  attachBtn: document.getElementById("attach-btn"),
  fileInput: document.getElementById("file-input"),
  cameraBtn: document.getElementById("camera-btn"),
  chatInput: document.getElementById("chat-input"),
  chatSend: document.getElementById("chat-send"),
  chatSendIcon: document.getElementById("chat-send-icon"),
  chatSendSpinner: document.getElementById("chat-send-spinner"),

  cameraPanel: document.getElementById("camera-panel"),
  cameraVideo: document.getElementById("camera-video"),
  cameraCanvas: document.getElementById("camera-canvas"),
  cameraCapture: document.getElementById("camera-capture"),
  cameraClose: document.getElementById("camera-close"),

  dropOverlay: document.getElementById("drop-overlay"),
};

let cameraStream = null;
let isSending = false;

// Dynamic show/hide uses inline styles rather than a Tailwind "hidden"
// class toggle — a static "hidden flex" combo let a later-emitted display
// utility win the cascade at the wrong time in an earlier version of this
// UI (see git history). Inline styles have unambiguous highest specificity,
// so this sidesteps that whole class of bug.
function setVisible(node, visible, display = "flex") {
  node.style.display = visible ? display : "none";
}

// ---------- Toasts ----------

function showToast(message, type = "info") {
  const styles = {
    error: "border-red-300 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200",
    success: "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-200",
    info: "border-slate-300 bg-white text-slate-800 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100",
  };
  const toast = document.createElement("div");
  toast.className = `animate-fade-in max-w-xs rounded-lg border px-4 py-3 text-sm shadow-lg ${styles[type] || styles.info}`;
  toast.textContent = message;
  el.toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), 4500);
}

// ---------- Wake-up ----------

async function bootstrap() {
  const ok = await waitForBackend({
    onAttempt: (attempt) => {
      el.wakeMessage.textContent =
        attempt <= 2
          ? "Connecting…"
          : "Waking up the server — this can take up to a minute on first use…";
    },
  });

  if (!ok) {
    el.wakeMessage.textContent = "Couldn't reach the server. Check your connection and reload the page.";
    return;
  }

  setVisible(el.wakeOverlay, false);
  setVisible(el.app, true, "flex");
}

// ---------- Attachments (tray) ----------

function processAttachment(record) {
  record.status = "processing";
  record.errorMessage = "";
  renderTray();

  record.processingPromise = processDocuments([record.file])
    .then(() => {
      if (!state.attachments.includes(record)) return;
      record.status = "done";
    })
    .catch((err) => {
      if (!state.attachments.includes(record)) return;
      record.status = "error";
      record.errorMessage = err instanceof ApiError ? err.message : "Failed to process.";
    })
    .finally(() => {
      if (state.attachments.includes(record)) renderTray();
    });
}

function addAttachment(file) {
  if (state.attachments.length >= MAX_TRAY_ATTACHMENTS) {
    showToast(`You can attach up to ${MAX_TRAY_ATTACHMENTS} images at once.`, "error");
    return;
  }
  const record = { id: crypto.randomUUID(), file, url: URL.createObjectURL(file), status: "processing", errorMessage: "" };
  state.attachments.push(record);
  processAttachment(record);
}

function removeAttachment(id) {
  const record = state.attachments.find((a) => a.id === id);
  state.attachments = state.attachments.filter((a) => a.id !== id);
  if (record) URL.revokeObjectURL(record.url);
  renderTray();
}

function retryAttachment(id) {
  const record = state.attachments.find((a) => a.id === id);
  if (record) processAttachment(record);
}

function buildTrayChip(record) {
  const chip = document.createElement("div");
  chip.className = "relative h-16 w-16 shrink-0 overflow-hidden rounded-lg border border-slate-200 dark:border-slate-600";

  const img = document.createElement("img");
  img.src = record.url;
  img.className = "h-full w-full object-cover";
  chip.appendChild(img);

  if (record.status === "processing") {
    const overlay = document.createElement("div");
    overlay.className = "absolute inset-0 flex items-center justify-center bg-black/50";
    const spin = document.createElement("span");
    spin.className = "spinner text-white";
    overlay.appendChild(spin);
    chip.appendChild(overlay);
  } else if (record.status === "error") {
    const overlay = document.createElement("div");
    overlay.className = "absolute inset-0 flex flex-col items-center justify-center gap-0.5 bg-red-900/80 text-white";
    const warn = document.createElement("span");
    warn.className = "text-base font-bold leading-none";
    warn.textContent = "!";
    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.className = "text-[10px] underline underline-offset-2";
    retryBtn.textContent = "Retry";
    retryBtn.title = record.errorMessage;
    retryBtn.addEventListener("click", () => retryAttachment(record.id));
    overlay.append(warn, retryBtn);
    chip.appendChild(overlay);
  } else {
    const badge = document.createElement("span");
    badge.className = "absolute bottom-1 right-1 flex h-4 w-4 items-center justify-center rounded-full bg-emerald-500 text-[10px] leading-none text-white";
    badge.textContent = "✓";
    chip.appendChild(badge);
  }

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.setAttribute("aria-label", "Remove attachment");
  removeBtn.className =
    "absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-slate-700 text-xs text-white shadow hover:bg-slate-900";
  removeBtn.textContent = "×";
  removeBtn.addEventListener("click", () => removeAttachment(record.id));
  chip.appendChild(removeBtn);

  return chip;
}

function renderTray() {
  setVisible(el.attachmentTray, state.attachments.length > 0, "flex");
  el.attachmentTray.replaceChildren();
  for (const record of state.attachments) {
    el.attachmentTray.appendChild(buildTrayChip(record));
  }
}

el.attachBtn.addEventListener("click", () => el.fileInput.click());

el.fileInput.addEventListener("change", (e) => {
  for (const file of e.target.files) {
    if (file.type.startsWith("image/")) addAttachment(file);
  }
  e.target.value = "";
});

// Drag-and-drop anywhere on the page attaches images, matching how chat
// apps typically handle it (no dedicated dropzone box).
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  e.preventDefault();
  dragDepth += 1;
  setVisible(el.dropOverlay, true, "flex");
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragleave", (e) => {
  e.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) setVisible(el.dropOverlay, false);
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  setVisible(el.dropOverlay, false);
  for (const file of e.dataTransfer.files) {
    if (file.type.startsWith("image/")) addAttachment(file);
  }
});

// ---------- Camera ----------

el.cameraBtn.addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    el.cameraVideo.srcObject = cameraStream;
    setVisible(el.cameraPanel, true, "flex");
  } catch {
    showToast("Couldn't access the camera. Check your browser permissions.", "error");
  }
});

function stopCamera() {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  setVisible(el.cameraPanel, false);
}

el.cameraClose.addEventListener("click", stopCamera);

el.cameraCapture.addEventListener("click", () => {
  const video = el.cameraVideo;
  const canvas = el.cameraCanvas;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    if (!blob) return;
    const file = new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" });
    addAttachment(file);
    showToast("Photo captured.", "success");
  }, "image/jpeg", 0.92);
});

// ---------- Chat ----------

function appendBubble({ role, text, attachments = [] }) {
  const row = document.createElement("div");
  row.className = `flex animate-fade-in ${role === "user" ? "justify-end" : "justify-start"}`;

  const bubble = document.createElement("div");
  bubble.className =
    role === "user"
      ? "max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-brand-600 px-4 py-2.5 text-sm text-white"
      : "max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-bl-sm bg-slate-100 px-4 py-2.5 text-sm leading-relaxed text-slate-800 dark:bg-slate-700 dark:text-slate-100";

  if (attachments.length > 0) {
    const thumbRow = document.createElement("div");
    thumbRow.className = "mb-2 flex flex-wrap gap-1.5";
    for (const a of attachments) {
      const link = document.createElement("a");
      link.href = a.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.title = "Open full size";
      const thumb = document.createElement("img");
      thumb.src = a.url;
      thumb.alt = a.file.name;
      thumb.className = "h-14 w-14 rounded-lg object-cover";
      link.appendChild(thumb);
      thumbRow.appendChild(link);
    }
    bubble.appendChild(thumbRow);
  }

  if (text) {
    const textEl = document.createElement("div");
    textEl.textContent = text;
    bubble.appendChild(textEl);
  }

  el.chatEmptyState?.remove();
  row.appendChild(bubble);
  el.chatLog.appendChild(row);
  el.chatLogWrap.scrollTop = el.chatLogWrap.scrollHeight;
  return { row, bubble };
}

function appendSources(container, sourcesText) {
  if (!sourcesText) return;
  const details = document.createElement("details");
  details.className = "mt-1.5 text-xs text-slate-500 dark:text-slate-400";
  const summary = document.createElement("summary");
  summary.className = "cursor-pointer select-none hover:text-slate-700 dark:hover:text-slate-300";
  summary.textContent = "Sources";
  const pre = document.createElement("pre");
  pre.className = "mt-1 whitespace-pre-wrap font-sans";
  pre.textContent = sourcesText;
  details.append(summary, pre);
  container.appendChild(details);
}

function setSendBusy(busy) {
  isSending = busy;
  el.chatSendIcon.classList.toggle("hidden", busy);
  el.chatSendSpinner.classList.toggle("hidden", !busy);
  el.chatSend.disabled = busy || !el.chatInput.value.trim();
}

el.chatInput.addEventListener("input", () => {
  if (!isSending) el.chatSend.disabled = !el.chatInput.value.trim();
});

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.chatInput.value.trim();
  if (!question || isSending) return;

  setSendBusy(true);

  const stillProcessing = state.attachments.filter((a) => a.status === "processing");
  if (stillProcessing.length > 0) {
    await Promise.all(stillProcessing.map((a) => a.processingPromise));
  }

  const turnAttachments = state.attachments.filter((a) => a.status === "done");
  const erroredCount = state.attachments.filter((a) => a.status === "error").length;
  if (erroredCount > 0) {
    showToast(`${erroredCount} attachment(s) failed to process and won't be included.`, "error");
  }
  // Bundled attachments leave the tray; errored ones stay for retry/removal.
  state.attachments = state.attachments.filter((a) => a.status !== "done");
  renderTray();

  appendBubble({ role: "user", text: question, attachments: turnAttachments });
  el.chatInput.value = "";
  el.chatInput.disabled = true;

  const { row: thinkingRow, bubble: thinkingBubble } = appendBubble({ role: "assistant", text: "" });
  thinkingBubble.classList.add("flex", "items-center", "gap-2");
  const dot = document.createElement("span");
  dot.className = "spinner text-slate-400";
  const label = document.createElement("span");
  label.textContent = "Thinking…";
  thinkingBubble.append(dot, label);

  try {
    const result = await sendChatMessage(question);
    thinkingRow.remove();
    const { bubble } = appendBubble({ role: "assistant", text: result.answer });
    if (result.is_grounded) appendSources(bubble, result.sources);
  } catch (err) {
    thinkingRow.remove();
    appendBubble({
      role: "assistant",
      text: err instanceof ApiError ? err.message : "Something went wrong answering that.",
    });
  } finally {
    el.chatInput.disabled = false;
    setSendBusy(false);
    el.chatInput.focus();
  }
});

// ---------- New chat ----------

el.newChatBtn.addEventListener("click", async () => {
  try {
    await resetDocuments();
  } catch {
    // best-effort; reset the UI regardless
  }
  for (const record of state.attachments) URL.revokeObjectURL(record.url);
  state.attachments = [];
  renderTray();
  el.chatLog.replaceChildren(el.chatEmptyState);
  showToast("Started a new chat.", "info");
});

bootstrap();
