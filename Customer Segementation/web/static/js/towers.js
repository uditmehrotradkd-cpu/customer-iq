// "Segment skyline": one glossy 3D tower per segment on a circular stage.
// Height = share of revenue, footprint = share of customers. Tap/click a tower to open its profile.
import { createStage, glowTexture, THREE } from "./stage3d.js";
import { reducedMotion } from "./effects.js";

function labelTexture(title, value) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 168;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
  ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(4, 4, 504, 160, 28);
  else ctx.rect(4, 4, 504, 160);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#ffffff";
  ctx.font = "700 66px Segoe UI, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(value, 256, 84);
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "600 40px Segoe UI, system-ui, sans-serif";
  ctx.fillText(title, 256, 138, 488);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export function mountTowers(container, profiles, colorOf, onSelect) {
  const stage = createStage(container, {
    label: "3D skyline: one tower per segment, height shows share of revenue. Drag to rotate, tap a tower to open it.",
    cameraPosition: [0, 7.2, 14.5],
    target: [0, 1.9, 0],
    shadows: true,
  });
  if (!stage) return () => {};
  const { scene, pivot, camera, el, state, onFrame } = stage;
  state.spin = reducedMotion ? 0 : 0.0035;
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
  const grid = new THREE.PolarGridHelper(5.9, 12, 6, 96, 0x22d3ee, 0x6366f1);
  grid.position.y = 0.012;
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  pivot.add(grid);

  // Rising "data" particles above the stage.
  const dustCount = reducedMotion ? 0 : 260;
  const dustPositions = new Float32Array(dustCount * 3);
  for (let i = 0; i < dustCount; i += 1) {
    const a = Math.random() * Math.PI * 2;
    const r = Math.sqrt(Math.random()) * 5.8;
    dustPositions.set([Math.cos(a) * r, Math.random() * 7, Math.sin(a) * r], i * 3);
  }
  const dustGeometry = new THREE.BufferGeometry();
  dustGeometry.setAttribute("position", new THREE.BufferAttribute(dustPositions, 3));

  const glow = glowTexture();
  stage.disposables.push(glow);
  if (dustCount) {
    pivot.add(new THREE.Points(dustGeometry, new THREE.PointsMaterial({ size: 0.09, map: glow, color: 0x67e8f9, transparent: true, opacity: 0.7, depthWrite: false, blending: THREE.AdditiveBlending })));
  }
  const towers = [];
  // Towers stand on a ring and the stage turns a full circle; alternating tall and short towers
  // keeps neighbouring labels at different heights so they don't collide.
  const byValue = [...profiles].sort((a, b) => b["Revenue_Share_%"] - a["Revenue_Share_%"]);
  const ordered = byValue.map((_, i) => (i % 2 === 0 ? byValue[i / 2] : byValue[byValue.length - 1 - (i - 1) / 2]));
  const n = ordered.length;
  const radius = n > 1 ? 3.6 : 0;
  // Scale to the tallest tower so one dominant segment still fits in view.
  const maxShare = Math.max(...profiles.map((p) => p["Revenue_Share_%"]), 1);
  ordered.forEach((p, i) => {
    const angle = (i / n) * Math.PI * 2;
    const width = Math.min(0.9 + (p["Customer_Share_%"] / 100) * 2.6, n > 1 ? radius * Math.sin(Math.PI / n) * 1.3 : 3);
    const height = 0.5 + (p["Revenue_Share_%"] / maxShare) * 5.2;
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
      emissiveIntensity: 0.28,
      transparent: true,
      opacity: 0.92,
    });
    const tower = new THREE.Mesh(geometry, material);
    tower.castShadow = true;
    tower.receiveShadow = true;
    tower.position.set(Math.cos(angle) * radius, 0, Math.sin(angle) * radius);
    tower.userData = { segment: p.Segment, height };
    tower.scale.y = reducedMotion ? 1 : 0.001;
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color: color.clone().lerp(new THREE.Color(0xffffff), 0.55), transparent: true, opacity: 0.9 }),
    );
    tower.add(edges);
    pivot.add(tower);

    const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
    halo.scale.setScalar(width * 2.2);
    halo.position.set(tower.position.x, 0.05, tower.position.z);
    pivot.add(halo);

    const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTexture(p.Segment_Name, `${p["Revenue_Share_%"].toFixed(1)}%`), depthWrite: false, depthTest: false, transparent: true }));
    label.renderOrder = 10;
    label.scale.set(2.3, 0.76, 1);
    label.position.set(tower.position.x, height + 0.8, tower.position.z);
    label.material.opacity = reducedMotion ? 1 : 0;
    pivot.add(label);

    towers.push({ tower, label, delay: i * 0.18 });
  });

  // Grow-in animation, a gentle hover bob on the labels and slowly rising particles.
  onFrame((t) => {
    for (let i = 1; i < dustCount * 3; i += 3) {
      dustPositions[i] = dustPositions[i] > 7 ? 0 : dustPositions[i] + 0.012;
    }
    if (dustCount) dustGeometry.attributes.position.needsUpdate = true;
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
