// "My account" page: profile, password, workspace status, recent chats and session.
import { api } from "./api.js";
import { downloadBlob, h, kpi } from "./dom.js";

const RECENT_CHATS = 5;

function initials(name) {
  const parts = name.split(" ").filter(Boolean);
  return ((parts[0]?.[0] ?? "") + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase() || "?";
}

function formatDate(value, options = { year: "numeric", month: "long", day: "numeric" }) {
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? "–" : d.toLocaleDateString(undefined, options);
}

function ago(ts) {
  const seconds = Math.round((ts - Date.now()) / 1000);
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, size] of [["day", 86400], ["hour", 3600], ["minute", 60]]) {
    if (Math.abs(seconds) >= size) return rtf.format(Math.round(seconds / size), unit);
  }
  return "just now";
}

function countDatasets(chats) {
  let total = 0;
  for (const chat of chats) {
    for (const m of chat.messages ?? []) {
      const res = m.role === "agent" ? m.res : null;
      if (res) total += (res.dataset?.kind ? 1 : 0) + (res.downloads ?? []).filter((d) => d.dataset?.kind).length;
    }
  }
  return total;
}

function status(el, message, ok = true) {
  el.textContent = message;
  el.classList.toggle("ok", ok);
  el.classList.toggle("bad", !ok);
}

function row(label, value) {
  return h("div", { class: "account-row" }, h("dt", {}, label), h("dd", {}, value));
}

function profileCard(user, onUserUpdated) {
  const name = h("input", { type: "text", maxlength: "40", value: user.name, autocomplete: "name", required: true });
  const note = h("p", { class: "form-status", role: "status" });
  const save = h("button", { class: "btn", type: "submit" }, "Save changes");
  const form = h(
    "form",
    { class: "account-form" },
    h("label", { class: "auth-field" }, h("span", {}, "Display name"), name),
    h("label", { class: "auth-field" }, h("span", {}, "Username"), h("input", { type: "text", value: user.username, readonly: true, "aria-readonly": "true" })),
    h("div", { class: "account-actions" }, save, note),
  );
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const value = name.value.trim();
    if (!value) return status(note, "Please enter a name.", false);
    save.disabled = true;
    try {
      const updated = await api.put("/auth/me", { name: value });
      name.value = updated.name;
      onUserUpdated(updated);
      status(note, "Saved.");
    } catch (err) {
      status(note, err.message, false);
    } finally {
      save.disabled = false;
    }
  });
  return h("section", { class: "card" }, h("h2", {}, "Profile"), h("p", { class: "caption" }, "How your name appears across the site."), form);
}

function passwordCard() {
  const input = (autocomplete) => h("input", { type: "password", maxlength: "128", autocomplete, required: true });
  const current = input("current-password");
  const next = input("new-password");
  const confirm = input("new-password");
  const note = h("p", { class: "form-status", role: "status" });
  const save = h("button", { class: "btn", type: "submit" }, "Update password");
  const form = h(
    "form",
    { class: "account-form" },
    h("label", { class: "auth-field" }, h("span", {}, "Current password"), current),
    h("label", { class: "auth-field" }, h("span", {}, "New password"), next),
    h("label", { class: "auth-field" }, h("span", {}, "Confirm new password"), confirm),
    h("div", { class: "account-actions" }, save, note),
  );
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (next.value.length < 8) return status(note, "The new password needs at least 8 characters.", false);
    if (next.value !== confirm.value) return status(note, "The new passwords don't match.", false);
    save.disabled = true;
    try {
      await api.post("/auth/me/password", { current_password: current.value, new_password: next.value });
      form.reset();
      status(note, "Password updated. Other devices have been signed out.");
    } catch (err) {
      status(note, err.message, false);
    } finally {
      save.disabled = false;
    }
  });
  return h("section", { class: "card" }, h("h2", {}, "Password"), h("p", { class: "caption" }, "Changing it signs you out everywhere else."), form);
}

function workspaceCard(ws, onModelChanged, rerender) {
  const own = ws.live_source === "published";
  const source = ws.mode === "generic" ? `Your dataset · ${ws.labels?.dataset ?? "uploaded file"}` : own ? "Your customer data" : "Demo customer data";
  const note = h("p", { class: "form-status", role: "status" });
  const report = h("button", { class: "btn secondary", type: "button" }, "Download report");
  report.addEventListener("click", async () => {
    report.disabled = true;
    try {
      downloadBlob(await api.postForBlob("/datasets", { kind: "report" }), "segment_report.xlsx");
    } catch (err) {
      status(note, err.message, false);
    } finally {
      report.disabled = false;
    }
  });
  const reset = own ? h("button", { class: "btn secondary", type: "button" }, "Reset to demo") : null;
  reset?.addEventListener("click", async () => {
    if (!window.confirm("Return your workspace to the original demo model? Your published model will be replaced.")) return;
    reset.disabled = true;
    try {
      await api.post("/agent/reset", {});
      onModelChanged();
      rerender();
    } catch (err) {
      status(note, err.message, false);
      reset.disabled = false;
    }
  });
  return h(
    "section",
    { class: "card" },
    h("h2", {}, "Workspace"),
    h("p", { class: "caption" }, "The model every page of the site uses for you."),
    h(
      "dl",
      { class: "account-list" },
      row("Source", source),
      row("Segments", h("span", { class: "account-chips" }, (ws.segments ?? []).map((s, i) => h("span", { class: "account-chip" }, h("span", { class: `dot dot-${i % 10}`, "aria-hidden": "true" }), s)))),
      row("Trained", ws.trained_at_utc ? formatDate(ws.trained_at_utc) : "–"),
      row("AI assistant", ws.ai?.enabled ? "On" : "Off"),
    ),
    h("div", { class: "account-actions" }, h("a", { class: "btn", href: "#/agent" }, "Upload new data"), report, reset, note),
  );
}

function chatsCard(chats) {
  const recent = [...chats].sort((a, b) => (b.updated ?? 0) - (a.updated ?? 0)).slice(0, RECENT_CHATS);
  return h(
    "section",
    { class: "card" },
    h("h2", {}, "Recent conversations"),
    recent.length
      ? h(
          "ul",
          { class: "account-chats" },
          recent.map((c) =>
            h("li", {}, h("a", { href: "#/agent" }, h("strong", {}, c.title), h("span", { class: "muted" }, `${c.updated ? ago(c.updated) : ""} · ${c.messages.length} messages`))),
          ),
        )
      : h("p", { class: "muted" }, "No conversations yet. Ask the data agent anything about your segments."),
    h("a", { class: "btn secondary", href: "#/agent" }, recent.length ? "Open chat history" : "Start a conversation"),
  );
}

export async function renderAccount(root, { user, onUserUpdated, onModelChanged, onSignOut, isCurrent }) {
  const [ws, history] = await Promise.all([api.fresh("/agent/status"), api.fresh("/auth/me/history").catch(() => ({ chats: [] }))]);
  if (!isCurrent()) return;
  const chats = (history.chats ?? []).filter((c) => c && typeof c.title === "string" && Array.isArray(c.messages));
  const rerender = () => renderAccount(root, { user, onUserUpdated, onModelChanged, onSignOut, isCurrent });

  const heading = h("h1", {}, user.name);
  const avatar = h("span", { class: "account-avatar", "aria-hidden": "true" }, initials(user.name));
  const updateUser = (updated) => {
    user = updated;
    heading.textContent = updated.name;
    avatar.textContent = initials(updated.name);
    onUserUpdated(updated);
  };

  root.replaceChildren(
    h(
      "section",
      { class: "banner plain account-hero" },
      avatar,
      h(
        "div",
        {},
        h("span", { class: "eyebrow" }, "My account"),
        heading,
        h("p", {}, `@${user.username} · Member since ${formatDate(user.created_at, { year: "numeric", month: "long" })}`),
      ),
    ),
    h(
      "div",
      { class: "kpis" },
      kpi("Workspace", ws.live_source === "published" ? "Your data" : "Demo"),
      kpi("Live segments", String(ws.n_segments)),
      kpi("Conversations", String(chats.length)),
      kpi("Datasets created", String(countDatasets(chats))),
    ),
    h("div", { class: "grid grid-2" }, profileCard(user, updateUser), passwordCard()),
    h("div", { class: "grid grid-5-7 account-lower" }, chatsCard(chats), workspaceCard(ws, onModelChanged, rerender)),
    h(
      "section",
      { class: "card account-signout" },
      h("div", {}, h("h2", {}, "Sign out"), h("p", { class: "caption" }, "End your session on this device. Your workspace and chats stay saved.")),
      h("button", { class: "btn secondary", type: "button", onclick: onSignOut }, "Sign out"),
    ),
  );
}
