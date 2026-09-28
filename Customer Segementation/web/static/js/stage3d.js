// Shared WebGL stage: renderer, camera, resize, drag-to-rotate, visibility-aware render loop and disposal.
import * as THREE from "../vendor/three.module.min.js";
import { reducedMotion } from "./effects.js";

export { THREE };

export function glowTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,0.85)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

export function createStage(container, { label, fov = 45, cameraPosition = [0, 0, 11], target = [0, 0, 0], shadows = false }) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" });
  } catch {
    container.classList.add("no-webgl");
    return null;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  if (shadows) {
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  }
  const el = renderer.domElement;
  el.setAttribute("role", "img");
  el.setAttribute("aria-label", label);
  container.append(el);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 200);
  camera.position.set(...cameraPosition);
  const lookAt = new THREE.Vector3(...target);
  camera.lookAt(lookAt);
  const pivot = new THREE.Group();
  scene.add(pivot);
  const disposables = [];

  const resize = () => {
    const { clientWidth: w, clientHeight: hgt } = container;
    if (!w || !hgt) return;
    renderer.setSize(w, hgt, false);
    camera.aspect = w / hgt;
    // Pull the camera back on narrow (portrait) screens so the whole scene stays in frame.
    camera.zoom = Math.min(1, camera.aspect / 1.1 + 0.25);
    camera.updateProjectionMatrix();
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();

  const state = { dragging: false, moved: 0, lastX: 0, lastY: 0, spin: reducedMotion ? 0 : 0.003, tiltLimit: 0.6 };
  const down = (e) => {
    state.dragging = true;
    state.moved = 0;
    state.lastX = e.clientX;
    state.lastY = e.clientY;
    el.setPointerCapture(e.pointerId);
  };
  const move = (e) => {
    if (!state.dragging) return;
    const dx = e.clientX - state.lastX;
    const dy = e.clientY - state.lastY;
    state.moved += Math.abs(dx) + Math.abs(dy);
    pivot.rotation.y += dx * 0.008;
    pivot.rotation.x = Math.max(-state.tiltLimit, Math.min(state.tiltLimit, pivot.rotation.x + dy * 0.004));
    state.lastX = e.clientX;
    state.lastY = e.clientY;
  };
  const up = () => {
    state.dragging = false;
  };
  el.addEventListener("pointerdown", down);
  el.addEventListener("pointermove", move);
  el.addEventListener("pointerup", up);
  el.addEventListener("pointercancel", up);

  let visible = true;
  const visibility = new IntersectionObserver(([entry]) => (visible = entry.isIntersecting));
  visibility.observe(container);

  const frameHandlers = [];
  let frame = 0;
  const clock = new THREE.Clock();
  const loop = () => {
    frame = requestAnimationFrame(loop);
    if (!visible || document.hidden) return;
    const t = clock.getElapsedTime();
    if (!state.dragging) pivot.rotation.y += state.spin;
    frameHandlers.forEach((fn) => fn(t));
    renderer.render(scene, camera);
  };
  loop();

  const dispose = () => {
    cancelAnimationFrame(frame);
    resizeObserver.disconnect();
    visibility.disconnect();
    scene.traverse((obj) => {
      obj.geometry?.dispose?.();
      const materials = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
      materials.forEach((m) => {
        m.map?.dispose?.();
        m.dispose();
      });
    });
    disposables.forEach((d) => d.dispose());
    renderer.dispose();
    el.remove();
  };

  return { renderer, scene, camera, pivot, el, state, disposables, onFrame: (fn) => frameHandlers.push(fn), dispose };
}
