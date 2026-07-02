// DocuMind SPA — talks to /upload, /chat (SSE), /api/sources, /api/summary.
// Runs as plain ES2017, no build step, no framework.

(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // ---------------- Theme persistence ---------------- //
  const THEME_KEY = "documind:theme";
  const storedTheme = localStorage.getItem(THEME_KEY) || "light";
  document.documentElement.classList.remove("light", "dark");
  document.documentElement.classList.add(storedTheme);
  const themeBtn = $("#themeToggle");
  themeBtn.textContent = storedTheme === "dark" ? "☀️ Light" : "🌙 Dark";
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.classList.contains("dark") ? "light" : "dark";
    document.documentElement.classList.remove("light", "dark");
    document.documentElement.classList.add(next);
    localStorage.setItem(THEME_KEY, next);
    themeBtn.textContent = next === "dark" ? "☀️ Light" : "🌙 Dark";
  });

  // ---------------- Optional admin token (sent if present) ---------------- //
  const TOKEN_KEY = "documind:token";
  const token = localStorage.getItem(TOKEN_KEY) || "";
  const authHeaders = token ? { "X-Documind-Token": token } : {};

  // ---------------- Tiny markdown renderer (safe, no eval) ---------------- //
  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
  function renderMarkdown(text) {
    const fenced = [];
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (m, lang, code) => {
      const i = fenced.push(`<pre><code>${escapeHtml(code)}</code></pre>`) - 1;
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
    const d = new Date();
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
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

  // ---------------- Suggestion chips ---------------- //
  const SUGGESTIONS = [
    "Summarize this document in a few sentences.",
    "What are the key points?",
    "List the main sections.",
    "What conclusions does it reach?",
  ];
  const suggestEl = $("#suggest");
  for (const q of SUGGESTIONS) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = q;
    b.addEventListener("click", () => {
      $("#askInput").value = q;
      $("#askForm").dispatchEvent(new Event("submit", { cancelable: true }));
    });
    suggestEl.appendChild(b);
  }

  // ---------------- Indexed docs + source scope ---------------- //
  let activeSource = null;
  const sourcePill = $("#sourcePill");
  const sourceName = $("#sourceName");
  $("#clearSource").addEventListener("click", () => setSource(null));

  function setSource(name) {
    activeSource = name;
    if (name) {
      sourceName.textContent = name;
      sourcePill.hidden = false;
    } else {
      sourcePill.hidden = true;
    }
  }

  async function refreshDocCount() {
    try {
      const r = await fetch("/api/sources", { headers: authHeaders });
      if (!r.ok) return;
      const data = await r.json();
      const n = (data.sources || []).length;
      $("#docCountN").textContent = String(n);
      $("#docCount").onclick = () => {
        if (n === 0) return;
        const pick = prompt(
          "Type the exact source name to scope questions to a single document:\n\n" +
            data.sources.join("\n") +
            "\n\n(leave blank to ask across all docs)",
          activeSource || ""
        );
        if (pick === null) return;
        setSource(pick.trim() || null);
      };
      $("#docCount").style.cursor = n ? "pointer" : "default";
    } catch (e) { /* network blip, ignore */ }
  }

  // ---------------- Upload ---------------- //
  const fileInput = $("#pdfFile");
  const status = $("#uploadStatus");
  fileInput.addEventListener("change", async () => {
    if (!fileInput.files || !fileInput.files[0]) return;
    await uploadFile(fileInput.files[0]);
    fileInput.value = "";
  });
  $("#serverPathBtn").addEventListener("click", async () => {
    const p = $("#serverPath").value.trim();
    if (!p) return;
    status.textContent = `Indexing ${p}…`;
    try {
      const r = await fetch("/api/upload-by-path", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders },
        body: JSON.stringify({ path: p }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      status.textContent = `Indexed ${data.processed.map((x) => x.source).join(", ")}`;
      enterChat(data.processed[0]?.source);
      await refreshDocCount();
    } catch (e) {
      status.innerHTML = `<span class="dm-err">${escapeHtml(String(e))}</span>`;
    }
  });

  async function uploadFile(file) {
    status.textContent = `Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)…`;
    const fd = new FormData();
    fd.append("files", file);
    try {
      const r = await fetch("/upload", { method: "POST", body: fd, headers: authHeaders });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      status.textContent = `Indexed ${data.processed.map((x) => x.source).join(", ")}`;
      enterChat(data.processed[0]?.source);
      await refreshDocCount();
    } catch (e) {
      status.innerHTML = `<span class="dm-err">${escapeHtml(String(e))}</span>`;
    }
  }

  // ---------------- Chat (SSE) ---------------- //
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

  function appendText(node, text) {
    const body = node.querySelector(".dm-msg-body");
    if (body.querySelector(".dm-thinking")) body.innerHTML = "";
    const raw = (body.dataset.raw || "") + text;
    body.dataset.raw = raw;
    body.innerHTML = renderMarkdown(raw);
    node.scrollIntoView({ block: "end", behavior: "smooth" });
  }

  askForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = askInput.value.trim();
    if (!q) return;
    enterChat();
    addMessage("user", q);
    askInput.value = "";
    askInput.style.height = "auto";
    sendBtn.disabled = true;

    const bot = addMessage("bot", "");
    const sources = [];

    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders },
        body: JSON.stringify({ question: q, source: activeSource, history: [] }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
        bot.querySelector(".dm-msg-body").innerHTML = `<span class="dm-err">${escapeHtml(err.error || "Request failed")}</span>`;
        return;
      }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const line = frame.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          let payload;
          try { payload = JSON.parse(line.slice(6)); } catch { continue; }
          if (payload.type === "sources") {
            sources.push(...(payload.sources || []));
            const det = bot.querySelector(".dm-sources");
            if (sources.length) {
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
          } else if (payload.type === "token") {
            appendText(bot, payload.text || "");
          } else if (payload.type === "error") {
            bot.querySelector(".dm-msg-body").innerHTML = `<span class="dm-err">${escapeHtml(payload.message || "Unknown error")}</span>`;
          }
        }
      }
    } catch (e) {
      bot.querySelector(".dm-msg-body").innerHTML = `<span class="dm-err">${escapeHtml(String(e))}</span>`;
    } finally {
      sendBtn.disabled = false;
    }
  });

  // ---------------- Admin token entry ---------------- //
  // The /upload-by-path and any future admin features need an X-Documind-Token
  // header. We let the user paste one in and store it client-side so reloads
  // keep it (the server doesn't see the token in plain HTTP, so this is the
  // standard "demo token" pattern: show it to users who need admin features).
  function renderAdmin() {
    const panel = $("#adminPanel");
    if (!token) {
      panel.innerHTML = `<button type="button" id="adminBtn" class="dm-admin-btn">Admin sign-in</button>`;
      $("#adminBtn").addEventListener("click", () => {
        const t = prompt("Paste your DocuMind admin token (sent in the X-Documind-Token header). Leave blank to clear.");
        if (t === null) return;
        localStorage.setItem(TOKEN_KEY, t.trim());
        location.reload();
      });
    } else {
      panel.innerHTML = `<button type="button" id="adminOut" class="dm-admin-btn">Sign out (admin)</button>`;
      $("#adminOut").addEventListener("click", () => {
        localStorage.removeItem(TOKEN_KEY);
        location.reload();
      });
    }
    // The path-upload box is only shown to admins because the server requires
    // the token to accept it; this is just UX.
    const pathBox = $("#pathAdmin");
    if (pathBox) pathBox.style.display = token ? "" : "none";
  }

  // ---------------- Boot ---------------- //
  renderAdmin();
  refreshDocCount();
  setInterval(refreshDocCount, 30000);
})();
