// "Customer galaxy": every customer is a glowing point in 3D PCA space, coloured by segment.
import { createStage, glowTexture, THREE } from "./stage3d.js";
import { reducedMotion } from "./effects.js";

export function mountUniverse(container, data, colorOf) {
  const stage = createStage(container, { label: "Rotating 3D map of all customers coloured by segment. Drag to rotate." });
  if (!stage) return () => {};
  const { pivot, onFrame } = stage;
  pivot.rotation.x = 0.35;

  const n = data.segments.length;
  const colors = new Float32Array(n * 3);
  const color = new THREE.Color();
  for (let i = 0; i < n; i += 1) {
    color.set(colorOf(data.segments[i]));
    colors.set([color.r, color.g, color.b], i * 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(data.positions), 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const offset = new THREE.Vector3();
  new THREE.Box3().setFromArray(data.positions).getCenter(offset);
  geometry.translate(-offset.x, -offset.y, -offset.z);

  const glow = glowTexture();
  stage.disposables.push(glow);
  pivot.add(
    new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ size: 0.22, map: glow, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.9 }),
    ),
  );

  const core = new THREE.SphereGeometry(0.28, 32, 32);
  const orbits = [];
  for (const c of data.centers) {
    const p = new THREE.Vector3(...c.position).sub(offset);
    const mesh = new THREE.Mesh(core, new THREE.MeshBasicMaterial({ color: colorOf(c.segment) }));
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color: colorOf(c.segment), transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
    halo.scale.setScalar(2.4);
    mesh.position.copy(p);
    halo.position.copy(p);
    const orbit = new THREE.Mesh(
      new THREE.TorusGeometry(0.75, 0.012, 8, 96),
      new THREE.MeshBasicMaterial({ color: colorOf(c.segment), transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    orbit.position.copy(p);
    orbit.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
    orbits.push({ orbit, halo, phase: Math.random() * Math.PI * 2 });
    pivot.add(mesh, halo, orbit);
  }

  // Distant starfield so the galaxy floats in space.
  const starCount = 900;
  const stars = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i += 1) {
    const v = new THREE.Vector3().randomDirection().multiplyScalar(18 + Math.random() * 22);
    stars.set([v.x, v.y, v.z], i * 3);
  }
  const starGeometry = new THREE.BufferGeometry();
  starGeometry.setAttribute("position", new THREE.BufferAttribute(stars, 3));
  stage.scene.add(new THREE.Points(starGeometry, new THREE.PointsMaterial({ size: 0.12, map: glow, color: 0xc7d2fe, transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending })));

  if (!reducedMotion) {
    onFrame((t) => {
      pivot.position.y = Math.sin(t) * 0.12;
      for (const { orbit, halo, phase } of orbits) {
        orbit.rotation.z = t * 0.8 + phase;
        orbit.rotation.x += 0.004;
        halo.scale.setScalar(2.4 + Math.sin(t * 2 + phase) * 0.35);
      }
    });
  }
  return stage.dispose;
}
