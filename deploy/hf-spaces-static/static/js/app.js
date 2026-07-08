// DocuMind static-space SPA — full client-side RAG.
//
// Pipeline: PDF -> PDF.js -> chunks -> embed (Xenova/all-MiniLM-L6-v2 in
// WASM) -> cosine search -> cross-encoder rerank (Xenova/cross-encoder-
// ms-marco-MiniLM-L-6-v2) -> prompt + sources -> Groq streaming chat -> SSE
// tokens -> markdown bubble with citations. IndexedDB persists the
// index between sessions.
//
// All five steps run in the browser. The only network call is the
// streaming chat completion to the LLM API (Groq by default, swappable
// to any OpenAI-compatible endpoint). No document text, embedding, or
// chunk ever leaves the device.

import { parsePdf } from "./pdf.js";
import { chunkPages } from "./chunker.js";
import {
  embedChunks,
  embedQuery,
  search,
  rerank,
} from "./rag-pipeline.js";
import { streamChat, getLlmConfig, setLlmConfig, hasLlmKey } from "./llm.js";
import { listDocs, saveDoc, deleteDoc, setMeta, getMeta } from "./store.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

// ---------------- Theme ---------------- //
(function theme() {
  const KEY = "documind:theme";
  const cur = localStorage.getItem(KEY) || "light";
  document.documentElement.classList.add(cur);
  const btn = $("#themeToggle");
  btn.textContent = cur === "dark" ? "☀️ Light" : "🌙 Dark";
  btn.addEventListener("click", () => {
    const next = document.documentElement.classList.contains("dark") ? "light" : "dark";
    document.documentElement.classList.remove("light", "dark");
    document.documentElement.classList.add(next);
    localStorage.setItem(KEY, next);
    btn.textContent = next === "dark" ? "☀️ Light" : "🌙 Dark";
  });
})();

// ---------------- Markdown renderer (safe, no eval) ---------------- //
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function renderMarkdown(text) {
  const fenced = [];
  text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, _l, c) => {
    const i = fenced.push(`<pre><code>${escapeHtml(c)}</code></pre>`) - 1;
    return `\u0000F${i}\u0000`;
  });
  text = escapeHtml(text);
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  text = text.replace(/\n{2,}/g, "</p><p>");
  text = `<p>${text.replace(/\n/g, "<br>")}</p>`;
  text = text.replace(/\u0000F(\d+)\u0000/g, (_, i) => fenced[Number(i)]);
  return text;
}

// ---------------- Helpers ---------------- //
function nowStr() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function makeBubble(role, text, sources) {
  const tpl = $("#bubbleTpl").content.cloneNode(true);
  const root = tpl.querySelector(".dm-msg");
  root.classList.add(role === "user" ? "from-user" : "from-bot");
  tpl.querySelector(".dm-msg-role").textContent = role === "user" ? "🧑 You" : "📄 DocuMind";
  tpl.querySelector(".dm-msg-time").textContent = nowStr();
  const body = tpl.querySelector(".dm-msg-body");
  body.innerHTML = text ? renderMarkdown(text) : '<span class="dm-thinking">thinking…</span>';
  if (sources && sources.length) {
    const det = tpl.querySelector(".dm-sources");
    det.hidden = false;
    const ol = tpl.querySelector(".dm-src-list");
    for (const s of sources) {
      const li = document.createElement("li");
      const page = typeof s.page === "number" ? ` · p.${s.page + 1}` : "";
      const rel = (s.score != null) ? ` <span class="src-rel">${Math.round(100 / (1 + Math.exp(-s.score)))}% match</span>` : "";
      const snip = s.snippet ? `<span class="src-snippet">${escapeHtml(s.snippet).slice(0, 280)}${s.snippet.length > 280 ? "…" : ""}</span>` : "";
      li.innerHTML = `<span class="src-head">${escapeHtml(s.source || "unknown")}${page}</span>${rel}${snip}`;
      ol.appendChild(li);
    }
  }
  return root;
}
function appendText(node, text) {
  const body = node.querySelector(".dm-msg-body");
  if (body.querySelector(".dm-thinking")) body.innerHTML = "";
  const raw = (body.dataset.raw || "") + text;
  body.dataset.raw = raw;
  body.innerHTML = renderMarkdown(raw);
  node.scrollIntoView({ block: "end", behavior: "smooth" });
}

// ---------------- Suggestion chips ---------------- //
const SUGGESTIONS = [
  "Summarize this document in a few sentences.",
  "What are the key points?",
  "List the main sections.",
  "What conclusions does it reach?",
];
for (const q of SUGGESTIONS) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = q;
  b.addEventListener("click", () => {
    $("#askInput").value = q;
    $("#askForm").dispatchEvent(new Event("submit", { cancelable: true }));
  });
  $("#suggest").appendChild(b);
}

// ---------------- State ---------------- //
let docs = []; // [{id, name, chunks, vectors}]
let activeSource = null;
let bootStatus = { embedder: false, reranker: false };

function setSource(name) {
  activeSource = name;
  if (name) {
    $("#sourceName").textContent = name;
    $("#sourcePill").hidden = false;
  } else {
    $("#sourcePill").hidden = true;
  }
}

function refreshDocCount() {
  const n = docs.length;
  $("#docCountN").textContent = String(n);
  $("#docCount").style.cursor = n ? "pointer" : "default";
  $("#docCount").onclick = () => {
    if (n === 0) return;
    const pick = prompt(
      "Type the exact source name to scope questions to a single document:\n\n" +
        docs.map((d) => d.name).join("\n") +
        "\n\n(leave blank to ask across all docs)",
      activeSource || ""
    );
    if (pick === null) return;
    setSource(pick.trim() || null);
  };
}

// ---------------- Admin / LLM key panel ---------------- //
function renderAdmin() {
  const panel = $("#adminPanel");
  if (!hasLlmKey()) {
    panel.innerHTML = `<button class="dm-admin-btn" id="adminBtn" type="button">Admin sign-in</button>`;
    $("#adminBtn").addEventListener("click", () => {
      const cfg = getLlmConfig();
      const key = prompt(
        "Paste a free Groq API key (get one at console.groq.com, no credit card).\n\n" +
          "Leave blank to clear.",
        cfg.apiKey
      );
      if (key === null) return;
      setLlmConfig({ apiKey: key.trim() });
      location.reload();
    });
  } else {
    panel.innerHTML = `<button class="dm-admin-btn" id="adminOut" type="button">Sign out</button>`;
    $("#adminOut").addEventListener("click", () => {
      if (!confirm("Sign out and clear the LLM key from this browser?")) return;
      setLlmConfig({ apiKey: "" });
      location.reload();
    });
  }
  // The path-upload box is only for server deployments. Hide it in
  // the static-space build — there's no server-side file system to
  // read from.
  const pathBox = $("#pathAdmin");
  if (pathBox) pathBox.style.display = "none";
}

// ---------------- Pipeline: PDF -> chunks -> vectors ---------------- //
async function ingestFile(file) {
  const status = $("#uploadStatus");
  status.textContent = `Reading ${file.name}…`;
  const pages = await parsePdf(file);
  status.textContent = `Chunking ${pages.length} page(s)…`;
  const chunks = chunkPages(pages, { chunkSize: 1000, chunkOverlap: 200 });
  if (chunks.length === 0) {
    status.innerHTML = `<span class="dm-err">${escapeHtml(file.name)}: no extractable text.</span>`;
    return null;
  }
  status.textContent = `Embedding ${chunks.length} chunks (first time loads ~50 MB of model weights)…`;
  bootStatus.embedder = true;
  const vectors = await embedChunks(chunks);
  // Persist to IndexedDB. Vectors are number[][], serialized natively.
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const doc = { id, name: file.name, chunks, vectors };
  await saveDoc(doc);
  docs.push(doc);
  return doc;
}

$("#pdfFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    const doc = await ingestFile(f);
    if (doc) {
      $("#uploadStatus").textContent = `Indexed ${doc.name} (${doc.chunks.length} chunks)`;
      enterChat(doc.name);
    }
  } catch (err) {
    $("#uploadStatus").innerHTML = `<span class="dm-err">${escapeHtml(String(err))}</span>`;
  }
  e.target.value = "";
});

// ---------------- Chat ---------------- //
const thread = $("#thread");
const chat = $("#chat");
const empty = $("#emptyState");
const askForm = $("#askForm");
const askInput = $("#askInput");
const sendBtn = $("#sendBtn");

askInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askForm.requestSubmit();
  }
});
askInput.addEventListener("input", () => {
  askInput.style.height = "auto";
  askInput.style.height = Math.min(askInput.scrollHeight, 180) + "px";
});

function enterChat(scope) {
  empty.hidden = true;
  chat.hidden = false;
  if (scope) setSource(scope);
}

function addMessage(role, text, sources) {
  const node = makeBubble(role, text, sources);
  thread.appendChild(node);
  node.scrollIntoView({ block: "end", behavior: "smooth" });
  return node;
}

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = askInput.value.trim();
  if (!q) return;
  if (docs.length === 0) {
    addMessage("bot", "Upload a PDF first, then ask. The whole RAG pipeline runs in your browser; nothing leaves the device except the LLM call.");
    return;
  }
  if (!hasLlmKey()) {
    addMessage("bot", '<span class="dm-err">Click "Admin sign-in" in the header to add a free Groq API key (no credit card). Until then, the LLM call can\'t be made.</span>');
    return;
  }
  enterChat();
  addMessage("user", q);
  askInput.value = "";
  askInput.style.height = "auto";
  sendBtn.disabled = true;

  const bot = addMessage("bot", "");
  const sources = [];

  try {
    // 1. Embed the question
    const qVec = await embedQuery(q);

    // 2. Filter chunks by active source if scoped
    const allChunks = [];
    const chunkDocIdx = []; // maps chunk -> doc index
    for (let di = 0; di < docs.length; di++) {
      if (activeSource && docs[di].name !== activeSource) continue;
      for (let ci = 0; ci < docs[di].chunks.length; ci++) {
        allChunks.push(docs[di].chunks[ci]);
        chunkDocIdx.push(di);
      }
    }
    if (allChunks.length === 0) {
      bot.querySelector(".dm-msg-body").innerHTML =
        `<span class="dm-err">No chunks under source "${escapeHtml(activeSource || "")}".</span>`;
      return;
    }
    const allVecs = [];
    for (let di = 0; di < docs.length; di++) {
      if (activeSource && docs[di].name !== activeSource) continue;
      for (const v of docs[di].vectors) allVecs.push(v);
    }

    // 3. Vector search top-30
    const candidates = search(qVec, allVecs, 30);
    for (const c of candidates) {
      sources.push({
        source: docs[chunkDocIdx[c.i]].name,
        page: allChunks[c.i].page,
        score: c.score,
        snippet: allChunks[c.i].text.slice(0, 400),
      });
    }
    renderSourcesInto(bot, sources);

    // 4. Cross-encoder re-rank top-12
    const ranked = await rerank(q, candidates, allChunks, 12);
    // Replace sources with the re-ranked order; preserve page + snippet.
    const newSources = ranked.map((r) => {
      const ch = allChunks[r.i];
      return {
        source: docs[chunkDocIdx[r.i]].name,
        page: ch.page,
        score: r.score,
        snippet: ch.text.slice(0, 400),
      };
    });
    sources.length = 0;
    sources.push(...newSources);
    renderSourcesInto(bot, sources);

    // 5. Build the prompt context from the re-ranked top-12.
    // Document-order sort: re-ranked list is relevance order, but the
    // model reads better when chunks appear in their natural page order.
    const docOrdered = [...ranked].sort((a, b) => {
      const ap = allChunks[a.i].page;
      const bp = allChunks[b.i].page;
      return ap - bp;
    });
    const context = docOrdered.map((r) => allChunks[r.i].text).join("\n\n");

    // 6. Stream the answer
    for await (const tok of streamChat({ context, question: q, history: [] })) {
      appendText(bot, tok);
    }
  } catch (err) {
    bot.querySelector(".dm-msg-body").innerHTML =
      `<span class="dm-err">${escapeHtml(String(err))}</span>`;
  } finally {
    sendBtn.disabled = false;
  }
});

function renderSourcesInto(bot, sources) {
  if (!sources || !sources.length) return;
  const det = bot.querySelector(".dm-sources");
  det.hidden = false;
  const ol = bot.querySelector(".dm-src-list");
  ol.innerHTML = "";
  for (const s of sources) {
    const li = document.createElement("li");
    const page = typeof s.page === "number" ? ` · p.${s.page + 1}` : "";
    const rel = (s.score != null) ? ` <span class="src-rel">${Math.round(100 / (1 + Math.exp(-s.score)))}% match</span>` : "";
    const snip = s.snippet ? `<span class="src-snippet">${escapeHtml(s.snippet).slice(0, 280)}${s.snippet.length > 280 ? "…" : ""}</span>` : "";
    li.innerHTML = `<span class="src-head">${escapeHtml(s.source || "unknown")}${page}</span>${rel}${snip}`;
    ol.appendChild(li);
  }
}

// ---------------- Boot ---------------- //
(async function init() {
  renderAdmin();
  docs = await listDocs();
  refreshDocCount();
  if (docs.length > 0) {
    $("#uploadStatus").textContent = `${docs.length} document(s) loaded from this browser. Drop another PDF to add to the index.`;
  }
})();
