// Data agent page: a personal chat workspace that answers questions, analyses / segments / scores / trains
// on uploaded files, publishes models and hands out datasets.
import { api } from "./api.js";
import { dataTable, downloadBlob, fmt, h, prettify } from "./dom.js";
import { backgroundVideo } from "./effects.js";
import { BAGS_VIDEO } from "./media.js";

const DRAFT_KEY = "cs-draft";
const MAX_CHATS = 30;
const MAX_MESSAGES = 150;
const AI_PROMPTS = [
  "Explain K-Means clustering in simple words",
  "Write a short marketing email for my largest segment",
  "What is customer lifetime value and how do I raise it?",
];
const MAX_MESSAGE = 2000;
const MAX_FILE_BYTES = 5 * 1024 * 1024;
const MAX_HISTORY_TURNS = 10;
const ACCEPT = ".csv,.tsv,.txt,.xlsx,.xlsm,.xls,.json,.jsonl,.ndjson";
const QUESTION_PROMPTS = [
  "Why this number of segments?",
  "What is the silhouette score?",
  "Tell me about digital deal-seekers",
  "Which segment has the highest income?",
  "Compare premium and affluent",
  "What file formats can I upload?",
];
const DATA_PROMPTS = [
  "Segment my own dataset",
  "Download the segment report with charts",
  "Give me premium customers with income over 70k",
  "Generate 1000 synthetic customers",
  "Download the recommendations table",
  "Train with 5 segments",
  "Upload template",
  "Status",
];
const GENERIC_DATA_PROMPTS = [
  "Segment my own dataset",
  "Download the segment report with charts",
  "Download the profiles table",
  "Compare all segments",
  "Which columns were used?",
  "Status",
  "Reset to original",
];
const STARTERS = [
  { icon: "spark", title: "Segment any dataset", text: "Drop a CSV, Excel or JSON file and I'll find segments using all of its columns.", prompt: "Segment my own dataset" },
  { icon: "chat", title: "Understand the segments", text: "Ask why the model chose its segments and what makes each one different.", prompt: "Why this number of segments?" },
  { icon: "table", title: "Pull customer lists", text: "Filter customers by segment, income, spend, recency and more.", prompt: "Give me premium customers with income over 70k" },
  { icon: "flask", title: "Generate synthetic data", text: "Realistic, privacy-safe customers that follow each segment's patterns.", prompt: "Generate 500 synthetic customers" },
];
const FILE_NAMES = {
  filtered: "customers_filtered.csv",
  synthetic: "synthetic_customers.csv",
  summary: "segment_table.csv",
  template: "customer_upload_template.csv",
  result: "scored_customers.csv",
  report: "segment_report.xlsx",
};
const FILE_ACTIONS = [
  { id: "auto", label: "Segment & apply", verb: "Segment and apply", thinking: "Reading your file, finding segments and rebuilding every page" },
  { id: "analyze", label: "Analyze only", verb: "Analyze", thinking: "Reading your file" },
  { id: "cluster", label: "Find segments (all columns)", verb: "Find segments in", thinking: "Finding segments across every column and rebuilding your workspace" },
  { id: "score", label: "Assign segments", verb: "Assign segments to", thinking: "Assigning every row to a segment" },
  { id: "train", label: "Train customer model", verb: "Train on", thinking: "Training a customer model on your file" },
];
const ACTION = Object.fromEntries(FILE_ACTIONS.map((a) => [a.id, a]));
const KIND_LABELS = { filtered: "Customer list", synthetic: "Synthetic customers", template: "Upload template", result: "Scored file", report: "Segment report (Excel)" };
const KEPT_FIELDS = ["reply", "intent", "applied", "ai_model", "dataset", "downloads", "preview", "segments_view", "draft", "links", "suggestions", "error"];

const ICONS = {
  clip: "M21.4 11.1 12.2 20.3a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5",
  send: "M4 12 20 4l-6 16-3-7-7-1Z",
  spark: "M12 3v4M12 17v4M3 12h4M17 12h4M6.3 6.3l2.8 2.8M14.9 14.9l2.8 2.8M6.3 17.7l2.8-2.8M14.9 9.1l2.8-2.8",
  chat: "M4 5h16v11H9l-5 4V5Z",
  table: "M4 5h16v14H4zM4 10h16M10 5v14",
  flask: "M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3",
  file: "M6 3h8l4 4v14H6zM14 3v4h4",
  download: "M12 4v11M7 10l5 5 5-5M5 20h14",
  refresh: "M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6",
  user: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM4 21a8 8 0 0 1 16 0",
  x: "M6 6l12 12M18 6 6 18",
  history: "M3 12a9 9 0 1 0 2.6-6.4M3 4v5h5M12 7v5l3 2",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
};

function icon(name, size = 18) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", ICONS[name]);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "currentColor");
  path.setAttribute("stroke-width", "2");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.append(path);
  return svg;
}

function storage(kind) {
  try {
    return kind === "local" ? window.localStorage : window.sessionStorage;
  } catch {
    return null;
  }
}

function read(kind, key) {
  try {
    return storage(kind)?.getItem(key) || null;
  } catch {
    return null;
  }
}

function write(kind, key, value) {
  try {
    value ? storage(kind)?.setItem(key, value) : storage(kind)?.removeItem(key);
  } catch {
    /* storage unavailable (private mode) */
  }
}

function initials(name) {
  const parts = name.split(" ").filter(Boolean);
  return ((parts[0]?.[0] ?? "") + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase() || "?";
}

function greeting() {
  const hour = new Date().getHours();
  return hour < 5 ? "Working late" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
}

function stamp(ts) {
  const d = new Date(ts);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return d.toDateString() === new Date().toDateString() ? time : `${d.toLocaleDateString([], { month: "short", day: "numeric" })}, ${time}`;
}

function ago(ts) {
  const seconds = Math.round((ts - Date.now()) / 1000);
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, size] of [["day", 86400], ["hour", 3600], ["minute", 60]]) {
    if (Math.abs(seconds) >= size) return rtf.format(Math.round(seconds / size), unit);
  }
  return "just now";
}

function validChats(data) {
  return Array.isArray(data) ? data.filter((c) => c && typeof c.id === "string" && typeof c.title === "string" && Array.isArray(c.messages)).slice(0, MAX_CHATS) : [];
}

function slim(res) {
  return Object.fromEntries(KEPT_FIELDS.filter((k) => res[k] != null).map((k) => [k, res[k]]));
}

function datasetTitle(dataset, label) {
  if (dataset.filename) return dataset.filename;
  if (label) return label;
  if (dataset.kind === "summary") {
    const name = prettify(dataset.table || "profiles");
    return `${name.charAt(0).toUpperCase()}${name.slice(1)} table`;
  }
  return KIND_LABELS[dataset.kind] ?? "Dataset";
}

function formatCell(value, column = "") {
  if (typeof value === "number" && /(^ID$|Year|Segment$)/.test(column)) return String(value);
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString("en-US") : fmt.number(value, 2);
  return value ?? "–";
}

export function renderAgent(root, { user, onModelChanged, onSignedOut }) {
  const userName = user.name;
  let draftId = read("session", DRAFT_KEY);
  let draftInfo = null;
  let busy = false;
  let lastFile = null;
  let fileAction = "auto";
  let history = [];
  let current = null;
  let historyTab = "chats";

  // ------------------------------------------------------------------ layout
  const heroTitle = h("h1", { class: "agent-title" });
  const statusPill = h("span", { class: "status-pill" }, h("span", { class: "pulse" }), "Connecting…");
  const account = h("span", { class: "account-pill" }, icon("user", 14), `Signed in as @${user.username}`);
  const heroVideo = backgroundVideo(BAGS_VIDEO, "background video");
  const eyebrow = h("span", { class: "agent-eyebrow" }, "CustomerIQ data agent");
  const hero = h(
    "section",
    { class: "agent-hero" },
    heroVideo.video,
    heroVideo.toggle,
    h("div", { class: "hero-blobs", "aria-hidden": "true" }, h("span"), h("span"), h("span")),
    h("div", { class: "agent-orb lg", "aria-hidden": "true" }, h("span", {}, "IQ")),
    h(
      "div",
      { class: "agent-hero-copy" },
      eyebrow,
      heroTitle,
      h("p", {}, "Ask about the segments, pull datasets, or drop in any CSV, Excel or JSON file and I'll segment it using every column."),
      h("div", { class: "hero-meta" }, statusPill, account),
    ),
  );

  const log = h("div", { class: "chat-log", role: "log", "aria-live": "polite", "aria-label": "Conversation with the data agent" });
  const input = h("textarea", { rows: "1", maxlength: String(MAX_MESSAGE), placeholder: "Message the data agent…", "aria-label": "Message" });
  const fileInput = h("input", { type: "file", accept: ACCEPT, class: "visually-hidden", id: "agent-file" });
  const fileChip = h("span", { class: "file-chip" });
  const actionBar = h("div", { class: "action-bar", role: "radiogroup", "aria-label": "What to do with the file" });
  const kSelect = h(
    "select",
    { "aria-label": "Number of segments", class: "k-select" },
    h("option", { value: "" }, "Auto segments"),
    [2, 3, 4, 5, 6, 7, 8, 9, 10].map((k) => h("option", { value: String(k) }, `${k} segments`)),
  );
  const fileTray = h("div", { class: "file-tray hidden" }, fileChip, actionBar, kSelect);
  const attach = h("label", { class: "round-btn", for: "agent-file", title: "Attach a CSV, Excel or JSON file" }, icon("clip"), h("span", { class: "visually-hidden" }, "Attach a file"));
  const sendLabel = h("span", { class: "send-label" }, "Send");
  const send = h("button", { class: "send-btn", type: "submit", "aria-label": "Send" }, icon("send", 17), sendLabel);
  const composer = h(
    "form",
    { class: "composer" },
    h("div", { class: "composer-box" }, fileTray, h("div", { class: "composer-row" }, fileInput, attach, input, send)),
    h("p", { class: "composer-hint" }, "Enter to send \u00b7 Shift+Enter for a new line \u00b7 CSV, Excel or JSON up to 5 MB"),
  );
  const dropHint = h("div", { class: "drop-hint", "aria-hidden": "true" }, icon("file", 34), h("strong", {}, "Drop your file"), h("span", {}, "CSV, Excel or JSON \u00b7 up to 5 MB"));
  const chatTitle = h("div", { class: "chat-title" }, h("strong", {}, "Data agent"), h("span", { class: "muted" }, "Online"));
  const historyCount = h("span", { class: "count-badge" });
  const historySearch = h("input", { type: "search", class: "history-search", placeholder: "Search chats and data", "aria-label": "Search history", maxlength: "80" });
  const historyList = h("div", { class: "history-list" });
  const historyTabs = h(
    "div",
    { class: "tabs", role: "tablist" },
    [
      ["chats", "Chats"],
      ["data", "Data"],
    ].map(([id, label]) =>
      h("button", { type: "button", role: "tab", "aria-selected": String(id === historyTab), dataset: { tab: id }, onclick: () => ((historyTab = id), renderHistory()) }, label),
    ),
  );
  const drawer = h(
    "aside",
    { class: "history-drawer", "aria-label": "Chat history" },
    h(
      "div",
      { class: "drawer-head" },
      h("div", {}, h("strong", {}, "History"), h("span", { class: "muted" }, "Saved to your account")),
      h("button", { type: "button", class: "chip-x", "aria-label": "Close history", onclick: () => toggleHistory(false) }, icon("x", 16)),
    ),
    historyTabs,
    historySearch,
    historyList,
    h("button", { type: "button", class: "ghost-link danger", onclick: clearHistory }, icon("trash", 15), "Clear all history"),
  );
  const scrim = h("div", { class: "drawer-scrim", onclick: () => toggleHistory(false) });
  const chatCard = h(
    "section",
    { class: "chat-shell" },
    h(
      "header",
      { class: "chat-head" },
      h("div", { class: "agent-orb sm", "aria-hidden": "true" }, h("span", {}, "IQ")),
      chatTitle,
      h("button", { type: "button", class: "ghost-link", onclick: () => toggleHistory(), title: "Previous chats and datasets", "aria-label": "History" }, icon("history", 15), "History", historyCount),
      h("button", { type: "button", class: "ghost-link", onclick: newChat, title: "Start a new conversation" }, icon("refresh", 15), "New chat"),
    ),
    log,
    composer,
    dropHint,
    scrim,
    drawer,
  );

  const draftPanel = h("div", { class: "draft-panel" });
  const promptList = (list) => h("div", { class: "prompt-list" }, list.map((p) => h("button", { type: "button", class: "prompt", onclick: () => submit(p) }, p)));
  const tabPanels = { ask: promptList(QUESTION_PROMPTS), data: promptList(DATA_PROMPTS) };
  const tabHost = h("div", { class: "tab-host" }, tabPanels.ask);
  const tabs = h(
    "div",
    { class: "tabs", role: "tablist" },
    [
      ["ask", "Ask"],
      ["data", "Data & training"],
    ].map(([id, label], i) =>
      h("button", { type: "button", role: "tab", "aria-selected": String(i === 0), onclick: (e) => selectTab(id, e.currentTarget) }, label),
    ),
  );
  function selectTab(id, button) {
    tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b === button)));
    tabHost.replaceChildren(tabPanels[id]);
  }

  root.replaceChildren(
    h(
      "div",
      { class: "agent-page" },
      hero,
      h(
        "div",
        { class: "agent-layout" },
        chatCard,
        h(
          "aside",
          { class: "agent-side" },
          h("section", { class: "side-card" }, h("h2", {}, "Model"), draftPanel),
          h("section", { class: "side-card" }, h("h2", {}, "Try asking"), tabs, tabHost),
          h(
            "section",
            { class: "side-card tips" },
            h("h2", {}, "Works with any file"),
            h(
              "ul",
              {},
              h("li", {}, h("strong", {}, "Find segments"), " works on any file: it uses every usable column and rebuilds the whole site (Overview, Profiles, Explorer, Assign, Methodology) from it."),
              h("li", {}, h("strong", {}, "Assign / Train"), " need the customer columns; the enrolment date is optional."),
              h("li", {}, "IDs, names, e-mails and phone numbers are never used to form segments."),
            ),
          ),
        ),
      ),
    ),
  );

  // ------------------------------------------------------------------ messages
  function scrollToEnd() {
    log.scrollTo({ top: log.scrollHeight, behavior: "smooth" });
  }

  function clearEmpty() {
    log.querySelector(".starter-grid")?.remove();
  }

  function message(role, body, at = Date.now()) {
    const avatar = role === "user" ? h("div", { class: "avatar user", "aria-hidden": "true" }, initials(userName)) : h("div", { class: "agent-orb xs", "aria-hidden": "true" }, h("span", {}, "IQ"));
    const who = role === "user" ? userName || "You" : "Data agent";
    return h("div", { class: `msg ${role}` }, avatar, h("div", { class: "msg-body" }, h("div", { class: "msg-meta" }, h("strong", {}, who), h("span", {}, stamp(at))), body));
  }

  function addUser(text, at) {
    clearEmpty();
    log.append(message("user", h("div", { class: "bubble" }, text), at));
    scrollToEnd();
  }

  function addThinking(label) {
    const el = message("agent", h("div", { class: "bubble typing" }, h("span", { class: "dots" }, h("i"), h("i"), h("i")), h("span", {}, label)));
    log.append(el);
    scrollToEnd();
    return el;
  }

  // A dataset is regenerated from the model it was built with, so remember that draft.
  function downloadButton(dataset, label = "Download CSV", primary = true, forDraft = draftId) {
    const { filename, ...payload } = dataset;
    const btn = h("button", { class: primary ? "btn" : "btn secondary", type: "button" }, icon("download", 16), label);
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        const blob = await api.postForBlob("/datasets", { ...payload, draft_id: forDraft || null });
        downloadBlob(blob, filename || (payload.kind === "summary" ? `segment_${payload.table}.csv` : FILE_NAMES[payload.kind]));
      } catch (err) {
        addAgent({ reply: `Download failed: ${err.message}` });
      } finally {
        btn.disabled = false;
      }
    });
    return btn;
  }

  function segmentCards(segments) {
    return h(
      "div",
      { class: "seg-grid" },
      segments.map((s, i) =>
        h(
          "article",
          { class: "seg-card", style: { animationDelay: `${i * 70}ms` } },
          h("span", { class: `seg-stripe stripe-${s.id % 10}` }),
          h("div", { class: "seg-top" }, h("span", { class: `dot dot-${s.id % 10}` }), h("h4", {}, s.name)),
          h("div", { class: "seg-share" }, h("strong", {}, `${fmt.number(s.share_pct, 1)}%`), s.rows != null ? h("span", { class: "muted" }, `${s.rows.toLocaleString("en-US")} rows`) : null),
          h("div", { class: "seg-bar" }, h("span", { class: `stripe-${s.id % 10}`, style: { width: `${Math.max(3, Math.min(100, s.share_pct))}%` } })),
          s.traits?.length ? h("ul", { class: "seg-traits" }, s.traits.map((t) => h("li", { class: t.direction || "" }, t.text))) : null,
        ),
      ),
    );
  }

  function addAgent(res, replace = null, { at, draft = draftId, restored = false } = {}) {
    const parts = [h("div", { class: "reply-text" }, res.reply)];
    const segments = res.segments_view
      ?? (res.intent === "trained" && res.draft
        ? res.draft.segments.map((s) => ({ id: s.id, name: s.name, share_pct: s.customers_pct, rows: null, traits: [{ text: `${s.revenue_pct.toFixed(0)}% of revenue \u00b7 avg spend ${fmt.currency(s.avg_spend)}` }] }))
        : null);
    if (segments?.length) parts.push(segmentCards(segments));
    if (res.preview && res.preview.columns.length) {
      const rows = res.preview.rows.map((r) => Object.fromEntries(res.preview.columns.map((c, i) => [c, r[i]])));
      parts.push(
        h(
          "details",
          { class: "preview", open: !segments?.length },
          h("summary", {}, `Preview \u00b7 ${Math.min(rows.length, res.preview.total)} of ${res.preview.total.toLocaleString("en-US")} rows`),
          dataTable(res.preview.columns.map((c) => ({ key: c, label: c, format: (v) => formatCell(v, c) })), rows),
        ),
      );
    }
    const actions = [];
    if (res.dataset) {
      const label = res.dataset.kind === "report" ? "Download report with charts (Excel)" : res.intent === "clustered" ? "Download labelled data" : "Download CSV";
      actions.push(downloadButton(res.dataset, label, true, draft));
    }
    (res.downloads || []).forEach((d) => actions.push(downloadButton(d.dataset, d.label, false, draft)));
    if (res.intent === "trained" && !res.applied && (!restored || res.draft?.draft_id === draftId)) actions.push(h("button", { class: "btn", type: "button", onclick: publish }, "Apply to my workspace"));
    if (!restored && lastFile && res.file_actions?.length) {
      const file = lastFile;
      res.file_actions.forEach((a, i) => actions.push(h("button", { class: i === 0 ? "btn" : "btn secondary", type: "button", onclick: () => sendFile(file, a) }, ACTION[a].label)));
    }
    (res.links || [])
      .filter(([, href]) => typeof href === "string" && /^(#\/|\/(?!\/))/.test(href))
      .forEach(([label, href]) => actions.push(h("a", { class: "btn secondary", href }, label)));
    if (actions.length) parts.push(h("div", { class: "msg-actions" }, actions));
    if (res.suggestions?.length) {
      parts.push(h("div", { class: "chips" }, res.suggestions.map((s) => h("button", { type: "button", class: "chip", onclick: () => submit(s) }, s))));
    }
    if (res.intent === "ai") {
      parts.push(h("div", { class: "ai-note" }, icon("spark", 13), `AI answer${res.ai_model ? ` · ${res.ai_model}` : ""} · can make mistakes, check important facts`));
    }
    const el = message("agent", h("div", { class: `bubble${res.error ? " error-bubble" : ""}` }, parts), at);
    replace ? replace.replaceWith(el) : log.append(el);
    scrollToEnd();
  }

  // ------------------------------------------------------------------ history (saved to the user's account)
  let saveTimer = null;

  function persist() {
    history = history.slice(0, MAX_CHATS);
    historyCount.textContent = history.length ? String(history.length) : "";
    if (chatCard.classList.contains("history-open")) renderHistory();
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveRemote, 500);
  }

  async function saveRemote() {
    for (;;) {
      try {
        await api.put("/auth/me/history", { chats: history });
        return;
      } catch (err) {
        if (err.status !== 413 || history.length <= 1) return;
        history.pop();
      }
    }
  }

  async function loadRemoteHistory() {
    try {
      const saved = validChats((await api.fresh("/auth/me/history")).chats);
      const known = new Set(history.map((c) => c.id));
      history = [...history, ...saved.filter((c) => !known.has(c.id))].slice(0, MAX_CHATS);
      historyCount.textContent = history.length ? String(history.length) : "";
      if (chatCard.classList.contains("history-open")) renderHistory();
    } catch {
      /* history stays empty until the next successful save */
    }
  }

  function record(entry) {
    if (!current) {
      const id = window.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;
      current = { id, title: entry.text.slice(0, 70), created: entry.at, updated: entry.at, messages: [] };
    }
    current.messages.push(entry);
    current.messages = current.messages.slice(-MAX_MESSAGES);
    current.updated = entry.at;
    history = [current, ...history.filter((c) => c.id !== current.id)];
    persist();
  }

  function openChat(conv) {
    if (busy) return;
    clearFile();
    current = conv;
    log.replaceChildren();
    for (const m of conv.messages) {
      try {
        if (m.role === "user" && typeof m.text === "string") addUser(m.text, m.at);
        else if (m.role === "agent" && typeof m.res?.reply === "string") addAgent(m.res, null, { at: m.at, draft: m.draft ?? null, restored: true });
      } catch {
        /* skip a malformed saved message */
      }
    }
    toggleHistory(false);
    log.scrollTo({ top: log.scrollHeight });
    input.focus();
  }

  function deleteChat(conv) {
    history = history.filter((c) => c.id !== conv.id);
    if (current?.id === conv.id) welcome();
    persist();
  }

  function clearHistory() {
    if (!history.length || !confirm("Delete all saved chats and dataset links from your account?")) return;
    history = [];
    welcome();
    persist();
  }

  function toggleHistory(open = !chatCard.classList.contains("history-open")) {
    chatCard.classList.toggle("history-open", open);
    if (open) {
      renderHistory();
      historySearch.focus();
    }
  }

  function matches(conv, query) {
    if (!query) return true;
    if (conv.title.toLowerCase().includes(query)) return true;
    return conv.messages.some((m) => String(m.text ?? m.res?.reply ?? "").toLowerCase().includes(query));
  }

  function renderHistory() {
    const query = historySearch.value.trim().toLowerCase();
    historyTabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === historyTab)));
    const empty = (text) => h("p", { class: "history-empty" }, text);

    if (historyTab === "chats") {
      const chats = history.filter((c) => matches(c, query));
      historyList.replaceChildren(
        ...(chats.length
          ? chats.map((c) =>
              h(
                "div",
                { class: `history-item${current?.id === c.id ? " active" : ""}` },
                h(
                  "button",
                  { type: "button", class: "history-link", onclick: () => openChat(c) },
                  h("span", { class: "history-icon" }, icon("chat", 16)),
                  h("span", { class: "history-text" }, h("strong", {}, c.title), h("span", {}, `${ago(c.updated)} · ${c.messages.length} messages`)),
                ),
                h("button", { type: "button", class: "chip-x", title: "Delete chat", "aria-label": `Delete chat ${c.title}`, onclick: () => deleteChat(c) }, icon("trash", 15)),
              ),
            )
          : [empty(query ? "No chats match your search." : "No chats yet. Your conversations will appear here.")]),
      );
      return;
    }

    const items = [];
    for (const conv of history) {
      for (const m of conv.messages) {
        const res = m.role === "agent" ? m.res : null;
        if (!res) continue;
        const found = [...(res.dataset ? [{ dataset: res.dataset, rows: res.preview?.total }] : []), ...(res.downloads || []).map((d) => ({ dataset: d.dataset, label: d.label }))];
        for (const f of found) {
          if (f.dataset?.kind) items.push({ ...f, conv, at: m.at, draft: m.draft ?? null, title: datasetTitle(f.dataset, f.label) });
        }
      }
    }
    const visible = items.filter((i) => !query || `${i.title} ${i.conv.title}`.toLowerCase().includes(query)).sort((a, b) => b.at - a.at);
    historyList.replaceChildren(
      ...(visible.length
        ? visible.map((i) =>
            h(
              "div",
              { class: "history-item data" },
              h("span", { class: "history-icon" }, icon("file", 16)),
              h(
                "span",
                { class: "history-text" },
                h("strong", {}, i.title),
                h("span", {}, [i.rows != null ? `${i.rows.toLocaleString("en-US")} rows` : null, ago(i.at)].filter(Boolean).join(" · ")),
                h("button", { type: "button", class: "link-btn", onclick: () => openChat(i.conv) }, `From: ${i.conv.title}`),
              ),
              downloadButton(i.dataset, "Download", false, i.draft),
            ),
          )
        : [empty(query ? "No datasets match your search." : "Datasets I create for you (lists, synthetic data, tables, scored and segmented files) will appear here.")]),
    );
  }

  historySearch.addEventListener("input", renderHistory);
  chatCard.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && chatCard.classList.contains("history-open")) toggleHistory(false);
  });

  function welcome() {
    current = null;
    log.replaceChildren();
    addAgent({
      reply:
        `Hi ${userName}! I'm your data agent. Ask me anything about the segments, the data or how the model works. ` +
        "Attach any CSV, Excel or JSON file and press Send: I find the segments and your whole workspace (Overview, Segment profiles, Explorer, Assign, Methodology) switches to that data. " +
        "Pick \"Analyze only\" to just inspect a file, or \"Assign segments\" to score rows with the current model.",
    });
    if (draftId) addAgent({ reply: "You still have a draft model open from earlier in this session. Datasets come from it until you publish or discard it." });
    log.append(
      h(
        "div",
        { class: "starter-grid" },
        STARTERS.map((s) =>
          h("button", { type: "button", class: "starter", onclick: () => (s.icon === "spark" ? fileInput.click() : submit(s.prompt)) }, h("span", { class: "starter-icon" }, icon(s.icon, 20)), h("strong", {}, s.title), h("span", {}, s.text)),
        ),
      ),
    );
  }

  function newChat() {
    if (busy) return;
    clearFile();
    welcome();
    input.focus();
  }

  // ------------------------------------------------------------------ model panel
  async function refreshPanel() {
    const live = await fetch("/api/v1/agent/status", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    const workspaceName = live ? (live.mode === "generic" ? live.labels?.dataset ?? "Your dataset" : live.live_source === "published" ? "Your data" : "Demo data") : "";
    statusPill.replaceChildren(
      h("span", { class: `pulse${live ? "" : " off"}` }),
      live ? `${workspaceName} \u00b7 ${live.n_segments} segments \u00b7 silhouette ${live.silhouette.toFixed(3)}` : "Model unavailable",
    );
    const generic = live?.mode === "generic";
    const aiOn = Boolean(live?.ai?.enabled);
    eyebrow.textContent = aiOn ? `CustomerIQ data agent \u00b7 AI answers by ${live.ai.model}` : "CustomerIQ data agent \u00b7 runs on this server, no external AI";
    input.placeholder = aiOn ? "Ask anything: your data, this site or any general question…" : "Message the data agent…";
    tabPanels.ask = promptList([...(generic && live.prompts?.length ? live.prompts : QUESTION_PROMPTS), ...(aiOn ? AI_PROMPTS : [])]);
    tabPanels.data = promptList(generic ? GENERIC_DATA_PROMPTS : DATA_PROMPTS);
    tabHost.replaceChildren(tabPanels[tabs.lastChild.getAttribute("aria-selected") === "true" ? "data" : "ask"]);
    const children = [];
    if (live) {
      children.push(
        h(
          "div",
          { class: "model-stats" },
          h("div", {}, h("span", {}, "Workspace"), h("strong", { title: workspaceName }, workspaceName)),
          h("div", {}, h("span", {}, "Segments"), h("strong", {}, String(live.n_segments))),
          h("div", {}, h("span", {}, "Silhouette"), h("strong", {}, live.silhouette.toFixed(3))),
        ),
      );
    }
    if (draftId && draftInfo) {
      children.push(
        h(
          "div",
          { class: "draft-box" },
          h("p", {}, h("strong", {}, "Your draft: "), `${draftInfo.k} segments \u00b7 ${draftInfo.rows_clean.toLocaleString("en-US")} customers \u00b7 silhouette ${draftInfo.silhouette.toFixed(3)}`),
          h("p", { class: "muted small" }, `Source: ${draftInfo.source}. Datasets you request now come from this draft.`),
          h(
            "ul",
            { class: "draft-segments" },
            draftInfo.segments.map((s) => h("li", {}, h("span", {}, h("span", { class: `dot dot-${s.id % 10}` }), s.name), h("span", { class: "muted" }, `${s.customers_pct.toFixed(0)}%`))),
          ),
          h("div", { class: "msg-actions" }, h("button", { class: "btn", type: "button", onclick: publish }, "Apply"), h("button", { class: "btn secondary", type: "button", onclick: discard }, "Discard")),
        ),
      );
    } else if (draftId) {
      children.push(h("p", { class: "muted small" }, "A draft is open for this session."), h("button", { class: "btn secondary", type: "button", onclick: discard }, "Discard draft"));
    } else {
      children.push(h("p", { class: "muted small" }, live?.live_source === "published" ? `Trained on your data${live.trained_at_utc ? ` \u00b7 ${new Date(live.trained_at_utc).toLocaleString()}` : ""}.` : "Upload a customer file and train to replace the demo data with yours."));
    }
    if (live?.live_source === "published") {
      children.push(h("button", { class: "btn secondary full", type: "button", onclick: () => submit("Reset to original") }, "Switch back to the demo data"));
    }
    draftPanel.replaceChildren(...children);
  }

  // ------------------------------------------------------------------ actions
  function toForm(obj) {
    const form = new FormData();
    Object.entries(obj).forEach(([k, v]) => v != null && form.append(k, v));
    return form;
  }

  async function publish() {
    if (!draftId) return;
    if (!confirm("Apply this model? Every page in your workspace will switch to its segments.")) return;
    await run("Apply to my workspace", () => api.post("/agent/publish", { draft_id: draftId }));
  }

  function discard() {
    draftId = null;
    draftInfo = null;
    write("session", DRAFT_KEY, null);
    addAgent({ reply: "Draft discarded. I'm using the live model again." });
    refreshPanel();
  }

  async function run(label, call, thinkingText = "Thinking") {
    if (busy) return;
    busy = true;
    send.disabled = true;
    addUser(label);
    record({ role: "user", text: label, at: Date.now() });
    const thinking = addThinking(thinkingText);
    try {
      const res = await call();
      if (res.intent === "trained" && res.draft && !res.applied) {
        draftId = res.draft.draft_id;
        draftInfo = res.draft;
        write("session", DRAFT_KEY, draftId);
      }
      if (res.applied || res.intent === "published" || res.intent === "reset_done") {
        if (res.intent !== "reset_done") {
          draftId = null;
          draftInfo = null;
          write("session", DRAFT_KEY, null);
        }
        api.clearCache();
        onModelChanged();
      }
      addAgent(res, thinking);
      record({ role: "agent", res: slim(res), at: Date.now(), draft: draftId });
      refreshPanel();
    } catch (err) {
      if (err.status === 401) {
        busy = false;
        onSignedOut?.();
        return;
      }
      if (err.status === 422 && /expired|Invalid draft/.test(err.message)) {
        draftId = null;
        write("session", DRAFT_KEY, null);
      }
      const failure = { reply: `Sorry, ${userName}: ${err.message}`, error: true };
      addAgent(failure, thinking);
      record({ role: "agent", res: failure, at: Date.now() });
    } finally {
      busy = false;
      send.disabled = false;
    }
  }

  function sendFile(file, action, k = null) {
    if (file.size > MAX_FILE_BYTES) return addAgent({ reply: "That file is larger than 5 MB.", error: true });
    lastFile = file;
    const withK = (action === "train" || action === "cluster" || action === "auto") && k;
    const label = `${ACTION[action].verb} ${file.name}${withK ? ` with ${k} segments` : ""}`;
    const fields = { file, action, k: withK ? k : null, draft_id: action === "score" ? draftId : null };
    return run(label, () => api.postForm("/agent/file", toForm(fields)), ACTION[action].thinking);
  }

  function submit(text) {
    const message = (text ?? input.value).trim();
    const file = fileInput.files[0];
    if (file && text == null) {
      const k = kSelect.value || null;
      const action = fileAction;
      clearFile();
      input.value = "";
      return sendFile(file, action, k);
    }
    if (!message) return;
    input.value = "";
    input.style.height = "auto";
    const training = /^\s*(please\s+)?(re-?train|rebuild|train)\b/i.test(message);
    return run(message, () => api.post("/agent/message", { message, draft_id: draftId, history: chatHistory() }), training ? "Training a new model, this takes a few seconds" : "Thinking");
  }

  // Earlier turns of this chat (excluding the message just sent) so the AI can follow up.
  function chatHistory() {
    return (current?.messages ?? [])
      .slice(0, -1)
      .map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: String(m.text ?? m.res?.reply ?? "").slice(0, 1500) }))
      .filter((m) => m.content)
      .slice(-MAX_HISTORY_TURNS);
  }

  // ------------------------------------------------------------------ file tray
  function renderActions() {
    actionBar.replaceChildren(
      ...FILE_ACTIONS.map((a) =>
        h("button", { type: "button", role: "radio", class: "action-pill", "aria-checked": String(a.id === fileAction), onclick: () => ((fileAction = a.id), syncFileControls()) }, a.label),
      ),
    );
  }

  function syncFileControls() {
    const hasFile = Boolean(fileInput.files[0]);
    fileTray.classList.toggle("hidden", !hasFile);
    renderActions();
    kSelect.classList.toggle("hidden", !["train", "cluster", "auto"].includes(fileAction));
    sendLabel.textContent = hasFile ? ACTION[fileAction].label.split(" (")[0] : "Send";
    composer.classList.toggle("has-file", hasFile);
  }

  function clearFile() {
    fileInput.value = "";
    syncFileControls();
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return clearFile();
    fileChip.replaceChildren(
      icon("file", 15),
      h("span", { class: "file-name" }, file.name),
      h("span", { class: "muted" }, `${Math.max(1, Math.round(file.size / 1024)).toLocaleString("en-US")} KB`),
      h("button", { type: "button", class: "chip-x", "aria-label": "Remove file", onclick: clearFile }, icon("x", 14)),
    );
    syncFileControls();
    input.focus();
  });

  let dragDepth = 0;
  chatCard.addEventListener("dragenter", (e) => {
    if (![...(e.dataTransfer?.types || [])].includes("Files")) return;
    e.preventDefault();
    dragDepth += 1;
    chatCard.classList.add("dragging");
  });
  chatCard.addEventListener("dragover", (e) => e.preventDefault());
  chatCard.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) chatCard.classList.remove("dragging");
  });
  chatCard.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    chatCard.classList.remove("dragging");
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    fileInput.dispatchEvent(new Event("change"));
  });

  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    submit();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
    input.style.overflowY = input.scrollHeight > 160 ? "auto" : "hidden";
  });

  // ------------------------------------------------------------------ start
  renderActions();
  syncFileControls();
  refreshPanel();
  heroTitle.replaceChildren(`${greeting()}, `, h("span", { class: "grad-text" }, userName));
  welcome();
  loadRemoteHistory();
}
