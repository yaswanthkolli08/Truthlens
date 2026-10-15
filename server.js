// TruthLens backend - the "kitchen" of your project 🍳
// It receives requests from index.html, asks the AI, and sends the answer back.
// Works with FREE Google Gemini (GEMINI_API_KEY) or with Claude (ANTHROPIC_API_KEY).

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 5000;

const GEMINI_KEY = (process.env.GEMINI_API_KEY || "").trim();
const CLAUDE_KEY = (process.env.ANTHROPIC_API_KEY || "").trim();
const GEMINI_OK = GEMINI_KEY !== "" && !GEMINI_KEY.startsWith("paste-");
const CLAUDE_OK = CLAUDE_KEY !== "" && !CLAUDE_KEY.startsWith("paste-");
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.8-flash";
// If the first model is busy, these backup models are tried one by one
const GEMINI_BACKUPS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash"];
const GEMINI_MODELS = [GEMINI_MODEL, ...GEMINI_BACKUPS.filter((m) => m !== GEMINI_MODEL)];
const CLAUDE_MODEL = process.env.MODEL || "claude-sonnet-5-5";

// Gemini is used first (it is free). If there is no Gemini key, Claude is used.
const PROVIDER = GEMINI_OK ? "gemini" : CLAUDE_OK ? "claude" : "none";
const MODEL = PROVIDER === "gemini" ? GEMINI_MODEL : CLAUDE_MODEL;

/* ---------- basic setup ---------- */
app.use(cors());
app.use(express.json({ limit: "25mb" })); // images are sent as text, so allow a big body
app.use(express.static(path.join(__dirname, "public"))); // serves index.html

/* ---------- simple rate limit: max 15 requests per minute per person ---------- */
const hits = new Map();
function rateLimit(req, res, next) {
  const now = Date.now();
  const recent = (hits.get(req.ip) || []).filter((t) => now - t < 60000);
  if (recent.length >= 15) {
    return res.status(429).json({ error: "Too many requests. Please wait a minute and try again." });
  }
  recent.push(now);
  hits.set(req.ip, recent);
  next();
}

/* ---------- helper: pull the JSON out of the AI's reply ---------- */
function extractJSON(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1) throw new Error("AI did not return JSON");
  return JSON.parse(text.slice(start, end + 1));
}

/* ---------- ask Google Gemini (free) ---------- */
// Tries one model. Returns { ok, status, data }
async function callGeminiModel(model, parts) {
  const url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent";
  const aiRes = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", "x-goog-api-key": GEMINI_KEY },
    body: JSON.stringify({
      contents: [{ role: "user", parts }],
      generationConfig: { responseMimeType: "application/json", maxOutputTokens: 8192, temperature: 0.3 },
    }),
  });
  let data = {};
  try { data = await aiRes.json(); } catch (e) {}
  return { ok: aiRes.ok, status: aiRes.status, data };
}

async function askGemini(prompt, images) {
  const parts = images.map((data) => ({ inline_data: { mime_type: "image/jpeg", data } }));
  parts.push({ text: prompt });

  let lastError = "Gemini returned an error.";

  // go through the models one by one until one of them answers
  for (const model of GEMINI_MODELS) {
    for (let attempt = 1; attempt <= 2; attempt++) {
      const r = await callGeminiModel(model, parts);

      if (r.ok) {
        if (r.data.promptFeedback && r.data.promptFeedback.blockReason) {
          throw new Error("Gemini blocked this content (" + r.data.promptFeedback.blockReason + ").");
        }
        const cand = r.data.candidates && r.data.candidates[0];
        const text = cand && cand.content && cand.content.parts
          ? cand.content.parts.map((p) => p.text || "").join("")
          : "";
        if (text) {
          console.log("✔ Answered by " + model);
          return text;
        }
        lastError = "Gemini sent an empty answer.";
        break; // try the next model
      }

      lastError = (r.data && r.data.error && r.data.error.message) || lastError;

      // wrong key or bad request: other models will not help
      if (r.status === 400 || r.status === 401 || r.status === 403) throw new Error(lastError);

      const busy = r.status === 429 || r.status === 500 || r.status === 503;
      if (busy && attempt === 1) {
        console.log("⏳ " + model + " is busy, trying once more...");
        await new Promise((res) => setTimeout(res, 1500));
        continue;
      }
      console.log("➡️  " + model + " did not work (" + r.status + "), trying the next model...");
      break;
    }
  }
  throw new Error(lastError);
}

/* ---------- ask Claude (needs credit) ---------- */
async function askClaude(prompt, images) {
  const content = images.map((data) => ({
    type: "image",
    source: { type: "base64", media_type: "image/jpeg", data },
  }));
  content.push({ type: "text", text: prompt });

  const aiRes = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": CLAUDE_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: CLAUDE_MODEL,
      max_tokens: 3000,
      messages: [{ role: "user", content }],
    }),
  });

  const data = await aiRes.json();
  if (!aiRes.ok) {
    throw new Error((data && data.error && data.error.message) || "Claude returned an error.");
  }
  return (data.content || []).map((b) => b.text || "").join("");
}

/* ---------- Route 1: health check (frontend asks "are you ready?") ---------- */
app.get("/api/health", (req, res) => {
  res.json({ ok: true, aiReady: PROVIDER !== "none", provider: PROVIDER, model: MODEL });
});

/* ---------- Route 2: analyze text or images with AI ---------- */
app.post("/api/analyze", rateLimit, async (req, res) => {
  try {
    if (PROVIDER === "none") {
      return res.status(500).json({ error: "AI key is missing. Add GEMINI_API_KEY in the .env file." });
    }

    const { prompt, images } = req.body || {};

    // check the input is safe and sensible
    if (typeof prompt !== "string" || !prompt.trim() || prompt.length > 30000) {
      return res.status(400).json({ error: "Invalid prompt." });
    }
    const imgs = Array.isArray(images) ? images : [];
    if (imgs.length > 4 || imgs.some((i) => typeof i !== "string")) {
      return res.status(400).json({ error: "You can send at most 4 images." });
    }

    const text = PROVIDER === "gemini" ? await askGemini(prompt, imgs) : await askClaude(prompt, imgs);
    res.json({ result: extractJSON(text) });
  } catch (err) {
    console.error("AI error:", err.message);
    res.status(502).json({ error: err.message });
  }
});

/* ---------- start ---------- */
app.listen(PORT, () => {
  console.log("✅ TruthLens running at http://localhost:" + PORT);
  if (PROVIDER === "none") console.log("⚠️  No AI key yet - add GEMINI_API_KEY in the .env file");
  else console.log("🤖 AI key found - using " + PROVIDER.toUpperCase() + " (" + MODEL + ")");
});
