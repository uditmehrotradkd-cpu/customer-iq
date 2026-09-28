// Site-wide animated backdrop: a slow "data network" of drifting nodes joined by faint lines.
// Skipped entirely in calm / reduced-motion mode; pauses while the tab is hidden.
import { reducedMotion } from "./effects.js";

const LINK_DISTANCE = 140;
const FRAME_MS = 1000 / 30;

export function startBackdrop(canvas = document.getElementById("fx-bg")) {
  if (!canvas || reducedMotion) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  let width = 0;
  let height = 0;
  let nodes = [];
  let color = "129, 140, 248";

  const readColor = () => {
    color = document.documentElement.getAttribute("data-theme") === "dark" ? "129, 140, 248" : "79, 70, 229";
  };

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const count = Math.min(70, Math.round((width * height) / 22000));
    nodes = Array.from({ length: count }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      r: 1 + Math.random() * 1.6,
    }));
  };

  let last = 0;
  const draw = (now) => {
    requestAnimationFrame(draw);
    if (document.hidden || now - last < FRAME_MS) return;
    last = now;
    ctx.clearRect(0, 0, width, height);
    for (const n of nodes) {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < -20) n.x = width + 20;
      if (n.x > width + 20) n.x = -20;
      if (n.y < -20) n.y = height + 20;
      if (n.y > height + 20) n.y = -20;
    }
    ctx.lineWidth = 1;
    for (let i = 0; i < nodes.length; i += 1) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j += 1) {
        const b = nodes[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < LINK_DISTANCE) {
          ctx.strokeStyle = `rgba(${color}, ${(1 - d / LINK_DISTANCE) * 0.22})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    ctx.fillStyle = `rgba(${color}, 0.55)`;
    for (const n of nodes) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  readColor();
  resize();
  window.addEventListener("resize", resize);
  new MutationObserver(readColor).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  requestAnimationFrame(draw);
}
