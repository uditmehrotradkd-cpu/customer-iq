// Light / Dark / Auto theme switcher. "auto" follows the operating-system setting live.
const KEY = "cs-theme";
const CHOICES = ["light", "dark", "auto"];
const media = window.matchMedia("(prefers-color-scheme: dark)");

function readPref() {
  try {
    const value = localStorage.getItem(KEY);
    return CHOICES.includes(value) ? value : "auto";
  } catch {
    return "auto";
  }
}

function savePref(value) {
  try {
    localStorage.setItem(KEY, value);
  } catch {
    /* preference simply won't persist */
  }
}

export function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

export function initThemeSwitch(onChange) {
  const root = document.documentElement;
  const buttons = [...document.querySelectorAll("[data-theme-choice]")];
  const meta = document.querySelector('meta[name="theme-color"]');
  let pref = readPref();

  const apply = () => {
    const resolved = pref === "dark" || (pref === "auto" && media.matches) ? "dark" : "light";
    const changed = root.getAttribute("data-theme") !== resolved;
    root.setAttribute("data-theme", resolved);
    root.setAttribute("data-theme-pref", pref);
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeChoice === pref)));
    if (meta) meta.setAttribute("content", resolved === "dark" ? "#111827" : "#ffffff");
    if (changed) onChange?.(resolved);
  };

  buttons.forEach((b) =>
    b.addEventListener("click", () => {
      pref = b.dataset.themeChoice;
      savePref(pref);
      apply();
    }),
  );
  media.addEventListener("change", () => pref === "auto" && apply());
  apply();
}
