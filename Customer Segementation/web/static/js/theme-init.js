// Runs before first paint (blocking, in <head>) so the page never flashes the wrong theme.
(function () {
  var pref = "auto";
  try {
    pref = localStorage.getItem("cs-theme") || "auto";
  } catch (e) {
    /* storage can be blocked (private mode); fall back to auto */
  }
  if (pref !== "light" && pref !== "dark") pref = "auto";
  var dark = pref === "dark" || (pref === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme-pref", pref);
})();
