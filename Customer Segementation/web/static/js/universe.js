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
  for (const c of data.centers) {
    const p = new THREE.Vector3(...c.position).sub(offset);
    const mesh = new THREE.Mesh(core, new THREE.MeshBasicMaterial({ color: colorOf(c.segment) }));
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color: colorOf(c.segment), transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
    halo.scale.setScalar(2.4);
    mesh.position.copy(p);
    halo.position.copy(p);
    pivot.add(mesh, halo);
  }

  if (!reducedMotion) onFrame((t) => (pivot.position.y = Math.sin(t) * 0.12));
  return stage.dispose;
}
