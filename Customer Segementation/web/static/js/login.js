// Sign-in / create-account screen. Sessions live in an HttpOnly cookie set by the server.
import { api } from "./api.js";
import { h } from "./dom.js";
import { backgroundVideo } from "./effects.js";
import { HERO_VIDEO } from "./media.js";

export async function fetchMe() {
  try {
    return await api.fresh("/auth/me");
  } catch {
    return null;
  }
}

export async function signOut() {
  try {
    await api.post("/auth/logout", {});
  } catch {
    /* the cookie is cleared server-side; ignore network errors */
  }
}

const FEATURES = [
  ["Your own workspace", "Upload your customer file and every page (overview, profiles, explorer, assignment) switches to your data."],
  ["Private by design", "Your models, scored files and chats belong to your account only."],
  ["Chat history", "Pick up any earlier conversation and re-download every dataset."],
];

export function renderLogin(root, onSignedIn) {
  let mode = "login";
  const nameInput = h("input", { type: "text", name: "name", autocomplete: "name", maxlength: "40", placeholder: "e.g. Priya Sharma" });
  const userInput = h("input", { type: "text", name: "username", autocomplete: "username", maxlength: "32", placeholder: "yourname", autocapitalize: "none", spellcheck: "false", required: true });
  const passInput = h("input", { type: "password", name: "password", autocomplete: "current-password", maxlength: "128", placeholder: "At least 8 characters", required: true });
  const showPass = h("button", { type: "button", class: "reveal-pass", "aria-label": "Show password", onclick: () => {
    const visible = passInput.type === "text";
    passInput.type = visible ? "password" : "text";
    showPass.textContent = visible ? "Show" : "Hide";
  } }, "Show");
  const nameField = h("label", { class: "auth-field" }, h("span", {}, "Your name"), nameInput);
  const errorLine = h("p", { class: "auth-error", role: "alert" });
  const submit = h("button", { class: "btn auth-submit", type: "submit" }, "Sign in");
  const title = h("h2", {}, "Welcome back");
  const subtitle = h("p", { class: "muted" }, "Sign in to open your customer intelligence workspace.");
  const tabs = h(
    "div",
    { class: "tabs auth-tabs", role: "tablist" },
    h("button", { type: "button", role: "tab", "aria-selected": "true", onclick: () => setMode("login") }, "Sign in"),
    h("button", { type: "button", role: "tab", "aria-selected": "false", onclick: () => setMode("register") }, "Create account"),
  );

  const form = h(
    "form",
    { class: "auth-form", novalidate: true },
    title,
    subtitle,
    tabs,
    nameField,
    h("label", { class: "auth-field" }, h("span", {}, "Username"), userInput),
    h("label", { class: "auth-field" }, h("span", {}, "Password"), h("div", { class: "pass-wrap" }, passInput, showPass)),
    errorLine,
    submit,
    h("p", { class: "auth-note" }, "Passwords are stored as salted scrypt hashes. Segments are for engagement planning, not credit or eligibility decisions."),
  );

  function setMode(next) {
    mode = next;
    const register = mode === "register";
    tabs.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-selected", String(i === (register ? 1 : 0))));
    nameField.classList.toggle("hidden", !register);
    title.textContent = register ? "Create your account" : "Welcome back";
    subtitle.textContent = register ? "It takes ten seconds. Your data stays in your own workspace." : "Sign in to open your customer intelligence workspace.";
    submit.textContent = register ? "Create account" : "Sign in";
    passInput.autocomplete = register ? "new-password" : "current-password";
    errorLine.textContent = "";
    (register ? nameInput : userInput).focus();
  }

  function validate() {
    const username = userInput.value.trim();
    if (mode === "register" && !nameInput.value.trim()) return "Please enter your name.";
    if (!/^[A-Za-z0-9_.-]{3,32}$/.test(username)) return "Username: 3–32 letters, numbers, dots, dashes or underscores.";
    if (passInput.value.length < 8) return "Password must be at least 8 characters.";
    return "";
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const problem = validate();
    errorLine.textContent = problem;
    if (problem) return;
    submit.disabled = true;
    const body = { username: userInput.value.trim(), password: passInput.value };
    if (mode === "register") body.name = nameInput.value.trim();
    try {
      const user = await api.post(mode === "register" ? "/auth/register" : "/auth/login", body);
      passInput.value = "";
      onSignedIn(user);
    } catch (err) {
      errorLine.textContent = err.message;
    } finally {
      submit.disabled = false;
    }
  });

  const { video, toggle } = backgroundVideo(HERO_VIDEO, "background video");
  root.replaceChildren(
    h(
      "div",
      { class: "auth-page" },
      h(
        "section",
        { class: "auth-hero" },
        video,
        toggle,
        h("div", { class: "hero-blobs", "aria-hidden": "true" }, h("span"), h("span"), h("span")),
        h("div", { class: "agent-orb lg", "aria-hidden": "true" }, h("span", {}, "IQ")),
        h("span", { class: "agent-eyebrow" }, "CustomerIQ"),
        h("h1", {}, "Turn your customer data into ", h("span", { class: "grad-text" }, "clear segments"), "."),
        h("ul", { class: "auth-features" }, FEATURES.map(([t, d]) => h("li", {}, h("strong", {}, t), h("span", {}, d)))),
      ),
      h("section", { class: "auth-card" }, form),
    ),
  );
  nameField.classList.add("hidden");
  userInput.focus();
}
