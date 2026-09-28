// Motion & depth effects: 3D tilt, scroll reveal, count-up numbers and background video helpers.
import { h } from "./dom.js";

// "Calm" (set by the motion toggle, applied before paint by theme-init.js) behaves like the OS reduced-motion setting.
export const calmMode = document.documentElement.getAttribute("data-motion") === "calm";
export const reducedMotion = calmMode || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const cleanups = [];

export function initMotionToggle() {
  const button = document.getElementById("motion-toggle");
  if (!button) return;
  button.setAttribute("aria-pressed", String(calmMode));
  button.addEventListener("click", () => {
    try {
      calmMode ? localStorage.removeItem("ciq-motion") : localStorage.setItem("ciq-motion", "calm");
    } catch {
      return;
    }
    location.reload();
  });
}

export function pageTransition(el) {
  if (reducedMotion) return;
  el.classList.remove("page-enter");
  void el.offsetWidth; // restart the CSS animation
  el.classList.add("page-enter");
}

export function onCleanup(fn) {
  cleanups.push(fn);
}

export function runCleanups() {
  while (cleanups.length) {
    try {
      cleanups.pop()();
    } catch {
      /* a failed cleanup must not block navigation */
    }
  }
}

export function tilt(el, { max = 10, scale = 1.02 } = {}) {
  if (reducedMotion || !window.matchMedia("(pointer: fine)").matches) return el;
  el.classList.add("tilt");
  const move = (e) => {
    const r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5;
    const y = (e.clientY - r.top) / r.height - 0.5;
    el.style.transform = `perspective(900px) rotateX(${(-y * max).toFixed(2)}deg) rotateY(${(x * max).toFixed(2)}deg) scale(${scale})`;
    el.style.setProperty("--glare-x", `${(x + 0.5) * 100}%`);
    el.style.setProperty("--glare-y", `${(y + 0.5) * 100}%`);
  };
  const leave = () => {
    el.style.transform = "";
  };
  el.addEventListener("pointermove", move);
  el.addEventListener("pointerleave", leave);
  return el;
}

let observer = null;
export function reveal(root) {
  const items = root.querySelectorAll(".card, .kpi, .coverflow, .reveal");
  if (reducedMotion || !("IntersectionObserver" in window)) {
    items.forEach((el) => el.classList.add("in-view"));
    return;
  }
  observer?.disconnect();
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          observer.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.08 },
  );
  items.forEach((el, i) => {
    el.classList.add("reveal-item");
    el.style.transitionDelay = `${Math.min(i % 6, 5) * 70}ms`;
    observer.observe(el);
  });
}

export function countUp(el, target, format, duration = 1200) {
  if (reducedMotion || !Number.isFinite(target)) {
    el.textContent = format(target);
    return;
  }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - (1 - t) ** 3;
    el.textContent = format(target * eased);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

export function backgroundVideo({ src, poster }, label) {
  const video = h("video", {
    class: "bg-video",
    poster,
    muted: true,
    loop: true,
    playsinline: true,
    preload: "metadata",
    "aria-hidden": "true",
    tabindex: "-1",
  });
  video.muted = true;
  video.append(h("source", { src, type: "video/mp4" }));
  if (!reducedMotion) {
    video.autoplay = true;
    video.addEventListener("canplay", () => video.play().catch(() => {}), { once: true });
  }
  const toggle = h("button", { class: "video-toggle", type: "button", "aria-label": `Pause ${label}` }, reducedMotion ? "▶ Play video" : "❚❚ Pause video");
  toggle.addEventListener("click", () => {
    if (video.paused) {
      video.play().catch(() => {});
      toggle.textContent = "❚❚ Pause video";
      toggle.setAttribute("aria-label", `Pause ${label}`);
    } else {
      video.pause();
      toggle.textContent = "▶ Play video";
      toggle.setAttribute("aria-label", `Play ${label}`);
    }
  });
  onCleanup(() => {
    video.pause();
    video.replaceChildren();
    video.load();
  });
  return { video, toggle };
}
