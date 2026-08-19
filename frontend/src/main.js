import { processDocuments, resetDocuments, sendChatMessage, waitForBackend, ApiError } from "./api.js";

const MAX_IMAGES = 20;

const state = {
  images: [], // { id, file } — pending, not yet sent to the backend
  documentsIndexed: false,
  indexedFilenames: [], // sent to the backend and indexed, across all "Process" clicks this session
  totalChunks: 0,
};

const el = {
  wakeOverlay: document.getElementById("wake-overlay"),
  wakeMessage: document.getElementById("wake-message"),
  app: document.getElementById("app"),
  toastContainer: document.getElementById("toast-container"),

  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  cameraBtn: document.getElementById("camera-btn"),
  cameraPanel: document.getElementById("camera-panel"),
  cameraVideo: document.getElementById("camera-video"),
  cameraCanvas: document.getElementById("camera-canvas"),
  cameraCapture: document.getElementById("camera-capture"),
  cameraClose: document.getElementById("camera-close"),
  imageCount: document.getElementById("image-count"),

  previewSection: document.getElementById("preview-section"),
  previewGrid: document.getElementById("preview-grid"),

  processSection: document.getElementById("process-section"),
  processBtn: document.getElementById("process-btn"),
  processBtnText: document.getElementById("process-btn-text"),
  processStatus: document.getElementById("process-status"),
  startOverBtn: document.getElementById("start-over-btn"),
  indexedDocs: document.getElementById("indexed-docs"),
  indexedDocsList: document.getElementById("indexed-docs-list"),

  qaSection: document.getElementById("qa-section"),
  chatLog: document.getElementById("chat-log"),
  chatEmptyState: document.getElementById("chat-empty-state"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  chatSend: document.getElementById("chat-send"),
};

let cameraStream = null;

// Applied together with removing "hidden" (never left sitting in the class
// list while "hidden" is also present) — a static "hidden lg:flex" combo
// would let lg:flex win the cascade at desktop widths even before qa-section
// is meant to be shown, since Tailwind's responsive utilities are emitted
// after the base "hidden" rule.
const QA_DESKTOP_LAYOUT_CLASSES = ["lg:sticky", "lg:top-8", "lg:flex", "lg:h-[calc(100vh-6rem)]", "lg:flex-col"];

function showQaSection() {
  el.qaSection.classList.remove("hidden");
  el.qaSection.classList.add(...QA_DESKTOP_LAYOUT_CLASSES);
}

function hideQaSection() {
  el.qaSection.classList.add("hidden");
  el.qaSection.classList.remove(...QA_DESKTOP_LAYOUT_CLASSES);
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

  el.wakeOverlay.classList.add("hidden");
  el.app.classList.remove("hidden");
}

// ---------- Image capture / upload ----------

function addFiles(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  const remaining = MAX_IMAGES - state.images.length;

  if (files.length > remaining) {
    showToast(`Only ${remaining} more image(s) can be added (max ${MAX_IMAGES}).`, "error");
  }

  for (const file of files.slice(0, remaining)) {
    state.images.push({ id: crypto.randomUUID(), file });
  }
  renderPreview();
}

function removeImage(id) {
  state.images = state.images.filter((img) => img.id !== id);
  renderPreview();
}

function renderPreview() {
  const count = state.images.length;
  el.imageCount.textContent = count ? `${count}/${MAX_IMAGES} image${count === 1 ? "" : "s"}` : "";
  el.previewSection.classList.toggle("hidden", count === 0);
  // Once something's been indexed this session, keep the Process section
  // visible (status + indexed-docs chips + Start Over) even with no new
  // pending images — it only fully hides before the first upload.
  el.processSection.classList.toggle("hidden", count === 0 && state.indexedFilenames.length === 0);
  el.processBtn.classList.toggle("hidden", count === 0);

  el.previewGrid.replaceChildren();
  for (const img of state.images) {
    const wrapper = document.createElement("div");
    wrapper.className = "group relative overflow-hidden rounded-lg border border-slate-200 dark:border-slate-700";

    const image = document.createElement("img");
    image.src = URL.createObjectURL(img.file);
    image.className = "aspect-square w-full object-cover";
    image.onload = () => URL.revokeObjectURL(image.src);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.setAttribute("aria-label", "Remove image");
    removeBtn.className =
      "absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-black/60 text-white opacity-0 transition group-hover:opacity-100 focus:opacity-100";
    removeBtn.textContent = "×";
    removeBtn.addEventListener("click", () => removeImage(img.id));

    wrapper.append(image, removeBtn);
    el.previewGrid.appendChild(wrapper);
  }
}

el.fileInput.addEventListener("change", (e) => {
  addFiles(e.target.files);
  e.target.value = "";
});

for (const evt of ["dragover", "dragenter"]) {
  el.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    el.dropzone.classList.add("border-brand-400", "bg-brand-50");
  });
}
for (const evt of ["dragleave", "dragend"]) {
  el.dropzone.addEventListener(evt, () => {
    el.dropzone.classList.remove("border-brand-400", "bg-brand-50");
  });
}
el.dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  el.dropzone.classList.remove("border-brand-400", "bg-brand-50");
  addFiles(e.dataTransfer.files);
});

// ---------- Camera ----------

el.cameraBtn.addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    el.cameraVideo.srcObject = cameraStream;
    el.cameraPanel.classList.remove("hidden");
  } catch {
    showToast("Couldn't access the camera. Check your browser permissions.", "error");
  }
});

function stopCamera() {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  el.cameraPanel.classList.add("hidden");
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
    addFiles([file]);
    showToast("Photo captured.", "success");
  }, "image/jpeg", 0.92);
});

function renderIndexedDocs() {
  el.indexedDocs.classList.toggle("hidden", state.indexedFilenames.length === 0);
  el.indexedDocsList.replaceChildren();
  for (const name of state.indexedFilenames) {
    const chip = document.createElement("li");
    chip.className = "rounded-full bg-brand-50 px-2.5 py-1 text-xs text-brand-700 dark:bg-brand-900/40 dark:text-brand-300";
    chip.textContent = name;
    el.indexedDocsList.appendChild(chip);
  }
}

// ---------- Process ----------

function setProcessing(isProcessing) {
  el.processBtn.disabled = isProcessing;
  el.processBtnText.textContent = isProcessing ? "" : "Process Documents";
  el.processBtn.querySelector(".spinner")?.remove();
  if (isProcessing) {
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    el.processBtn.prepend(spinner);
    el.processBtnText.textContent = "Processing…";
  }
}

el.processBtn.addEventListener("click", async () => {
  if (state.images.length === 0) return;
  setProcessing(true);
  el.processStatus.textContent = "";
  try {
    const summary = await processDocuments(state.images.map((img) => img.file));

    state.indexedFilenames.push(...state.images.map((img) => img.file.name));
    state.totalChunks += summary.chunks_indexed;
    renderIndexedDocs();

    // Clear the pending list now that these are indexed server-side —
    // otherwise the next "Process" click would resend (and re-index) them.
    state.images = [];
    renderPreview();

    el.processStatus.textContent = `Indexed ${summary.chunks_indexed} chunk(s) from ${summary.documents_processed} document(s) just now — ${state.totalChunks} chunk(s) total this session.`;
    state.documentsIndexed = true;
    showQaSection();
    el.startOverBtn.classList.remove("hidden");
    el.chatInput.focus();
  } catch (err) {
    showToast(err instanceof ApiError ? err.message : "Failed to process documents.", "error");
  } finally {
    setProcessing(false);
  }
});

el.startOverBtn.addEventListener("click", async () => {
  try {
    await resetDocuments();
  } catch {
    // best-effort; reset the UI regardless
  }
  state.images = [];
  state.documentsIndexed = false;
  state.indexedFilenames = [];
  state.totalChunks = 0;
  renderIndexedDocs();
  renderPreview();
  el.chatLog.replaceChildren(el.chatEmptyState);
  el.processStatus.textContent = "";
  hideQaSection();
  el.startOverBtn.classList.add("hidden");
  showToast("Started a new session.", "info");
});

// ---------- Chat ----------

function appendBubble({ role, text }) {
  const row = document.createElement("div");
  row.className = `flex animate-fade-in ${role === "user" ? "justify-end" : "justify-start"}`;

  const bubble = document.createElement("div");
  bubble.className =
    role === "user"
      ? "max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-brand-600 px-4 py-2.5 text-sm text-white"
      : "max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-bl-sm bg-slate-100 px-4 py-2.5 text-sm leading-relaxed text-slate-800 dark:bg-slate-700 dark:text-slate-100";
  bubble.textContent = text;

  el.chatEmptyState?.remove();
  row.appendChild(bubble);
  el.chatLog.appendChild(row);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
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

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.chatInput.value.trim();
  if (!question) return;

  appendBubble({ role: "user", text: question });
  el.chatInput.value = "";
  el.chatInput.disabled = true;
  el.chatSend.disabled = true;

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
    el.chatSend.disabled = false;
    el.chatInput.focus();
  }
});

bootstrap();
