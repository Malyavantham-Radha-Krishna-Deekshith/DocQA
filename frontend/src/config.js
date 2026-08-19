// Backend base URL. Auto-detects localhost for dev; edit RENDER_API_URL to
// your deployed Render service URL before deploying to Vercel.
const RENDER_API_URL = "https://docqa-hlwk.onrender.com";

const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);

export const API_BASE_URL = isLocal ? "http://localhost:8000" : RENDER_API_URL;
