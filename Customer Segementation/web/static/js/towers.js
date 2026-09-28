// "Segment skyline": one glossy 3D tower per segment on a circular stage.
// Height = share of revenue, footprint = share of customers. Tap/click a tower to open its profile.
import { createStage, glowTexture, THREE } from "./stage3d.js";
import { reducedMotion } from "./effects.js";

function labelTexture(title, value) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 160;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(15, 23, 42, 0.78)";
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(4, 4, 504, 152, 28);
  else ctx.rect(4, 4, 504, 152);
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.font = "700 64px Segoe UI, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(value, 256, 82);
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "500 30px Segoe UI, system-ui, sans-serif";
  ctx.fillText(title, 256, 128, 480);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export function mountTowers(container, profiles, colorOf, onSelect) {
  const stage = createStage(container, {
    label: "3D skyline: one tower per segment, height shows share of revenue. Drag to rotate, tap a tower to open it.",
    cameraPosition: [0, 7.5, 14.5],
    target: [0, 1.4, 0],
    shadows: true,
  });
  if (!stage) return () => {};
  const { scene, pivot, camera, el, state, onFrame } = stage;
  state.spin = reducedMotion ? 0 : 0.0025;
  state.tiltLimit = 0.25;

  scene.add(new THREE.HemisphereLight(0xdbeafe, 0x0f172a, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(5, 10, 6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8 });
  scene.add(sun);
  const rim = new THREE.PointLight(0xa5b4fc, 30, 30);
  rim.position.set(-6, 4, -5);
  scene.add(rim);

  const floor = new THREE.Mesh(
    new THREE.CylinderGeometry(6, 6.3, 0.35, 96),
    new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.55, metalness: 0.2 }),
  );
  floor.position.y = -0.18;
  floor.receiveShadow = true;
  pivot.add(floor);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(6.05, 0.03, 12, 160),
    new THREE.MeshBasicMaterial({ color: 0xa5b4fc, transparent: true, opacity: 0.6 }),
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.01;
  pivot.add(ring);

  const glow = glowTexture();
  stage.disposables.push(glow);
  const towers = [];
  const n = profiles.length;
  // Scale to the tallest tower so one dominant segment still fits in view.
  const maxShare = Math.max(...profiles.map((p) => p["Revenue_Share_%"]), 1);
  profiles.forEach((p, i) => {
    const angle = (i / n) * Math.PI * 2;
    const radius = n > 1 ? 3.3 : 0;
    const height = 0.5 + (p["Revenue_Share_%"] / maxShare) * 5.5;
    const width = 0.9 + (p["Customer_Share_%"] / 100) * 3.2;
    const color = new THREE.Color(colorOf(p.Segment));

    const geometry = new THREE.BoxGeometry(width, height, width);
    geometry.translate(0, height / 2, 0);
    const material = new THREE.MeshPhysicalMaterial({
      color,
      roughness: 0.25,
      metalness: 0.1,
      clearcoat: 1,
      clearcoatRoughness: 0.2,
      emissive: color,
      emissiveIntensity: 0.12,
    });
    const tower = new THREE.Mesh(geometry, material);
    tower.castShadow = true;
    tower.receiveShadow = true;
    tower.position.set(Math.cos(angle) * radius, 0, Math.sin(angle) * radius);
    tower.userData = { segment: p.Segment, height };
    tower.scale.y = reducedMotion ? 1 : 0.001;
    pivot.add(tower);

    const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
    halo.scale.setScalar(width * 2.2);
    halo.position.set(tower.position.x, 0.05, tower.position.z);
    pivot.add(halo);

    const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTexture(p.Segment_Name, `${p["Revenue_Share_%"].toFixed(1)}%`), depthWrite: false, transparent: true }));
    label.scale.set(2.6, 0.81, 1);
    label.position.set(tower.position.x, height + 0.8, tower.position.z);
    label.material.opacity = reducedMotion ? 1 : 0;
    pivot.add(label);

    towers.push({ tower, label, delay: i * 0.18 });
  });

  // Grow-in animation, then a gentle hover bob on the labels.
  onFrame((t) => {
    for (const { tower, label, delay } of towers) {
      const k = reducedMotion ? 1 : Math.min(Math.max((t - delay) / 1.2, 0), 1);
      const eased = 1 - (1 - k) ** 3;
      tower.scale.y = Math.max(eased, 0.001);
      label.position.y = tower.userData.height * eased + 0.8 + (reducedMotion ? 0 : Math.sin(t * 1.5 + delay * 10) * 0.08);
      label.material.opacity = eased;
    }
  });

  // A click that was not a drag selects the tower under the pointer.
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const meshes = towers.map((x) => x.tower);
  const pick = (e) => {
    const r = el.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    return raycaster.intersectObjects(meshes)[0]?.object ?? null;
  };
  el.addEventListener("pointerup", (e) => {
    if (state.moved > 6) return;
    const hit = pick(e);
    if (hit) onSelect(hit.userData.segment);
  });
  el.addEventListener("pointermove", (e) => {
    if (state.dragging) return;
    el.style.cursor = pick(e) ? "pointer" : "grab";
  });

  return stage.dispose;
}
