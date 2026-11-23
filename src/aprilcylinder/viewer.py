"""Interactive 3D viewer (single HTML file, three.js) and OBJ export for CAD tools."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .synth import load_painted_mesh

THREE = "https://cdn.jsdelivr.net/npm/three@0.160.0"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _part(V: np.ndarray, F: np.ndarray, white: np.ndarray | None, joint: str, solid: str | None = None) -> dict:
    return {"v": _b64(V.astype(np.float32)), "i": _b64(F.ravel().astype(np.uint16 if len(V) < 65536 else np.uint32)),
            "i32": len(V) >= 65536, "w": _b64(white.astype(np.uint8)) if white is not None else None,
            "solid": solid, "joint": joint}


def fixture_from_stl(stl_path, up_axis: str = "z", axis_xy=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Load a fixed part and move it so that the rotation axis (axis_xy, after re-orienting) is at x = y = 0."""
    from .mesh import Mesh, load_stl, reorient_up
    tris = reorient_up(load_stl(stl_path), up_axis) - np.array([axis_xy[0], axis_xy[1], 0.0])
    m = Mesh()
    for t in tris:
        m.tri(t[0], t[1], t[2], np.cross(t[1] - t[0], t[2] - t[0]))
    return np.asarray(m.verts), np.array([t[:3] for t in m.tris], dtype=np.int64)


def demo_fixture(plate_mm: float = 64.0, plate_t: float = 8.0, post_d: float = 15.0, post_h: float = 100.0):
    """A generic printed base: square plate with a round post on the axis. Returns (V, F), seat height."""
    from .cylinder import Section, TargetSpec, build
    spec = TargetSpec(sections=[Section("polygon", plate_t, sides=4, apothem=plate_mm / 2, markers=False),
                                Section("cylinder", post_h, radius=post_d / 2)],
                      top_marker=False, bottom_marker=False, segments=64)
    mesh, _, _ = build(spec, allow_no_markers=True)
    return (np.asarray(mesh.verts), np.array([t[:3] for t in mesh.tris], dtype=np.int64)), plate_t


def scene_payload(model_dir, label: str | None = None, fixture=None, seat_z: float = 0.0,
                  fixture_name: str | None = None) -> dict:
    """One viewer scene: the target (revolute about Z) and optionally a fixed part.

    fixture: (V, F) arrays of the fixed part in the target frame (axis at x = y = 0), or None.
    seat_z: height of the target's bottom face above z = 0 of the fixture.
    """
    model_dir = Path(model_dir)
    V, T = load_painted_mesh(model_dir)
    V = V + np.array([0.0, 0.0, seat_z])
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    F = np.array([t[:3] for t in T], dtype=np.int64)
    white = np.array([1 if t[3] else 0 for t in T], dtype=np.uint8)
    parts = [_part(V, F, white, "revolute")]
    allV = [V]
    if fixture is not None:
        fV, fF = fixture
        parts.insert(0, _part(np.asarray(fV, float), np.asarray(fF), None, "fixed", solid="fixture"))
        allV.append(np.asarray(fV, float))
    P = np.vstack(allV)
    lo, hi = P.min(0), P.max(0)
    markers = []
    for mk in cfg.get("markers", []):
        c = np.asarray(mk["corners_mm"], float).mean(0) + np.array([0.0, 0.0, seat_z])
        markers.append({"id": int(mk["id"]), "c": np.round(c, 3).tolist(),
                        "n": np.round(np.asarray(mk["normal"], float), 4).tolist(),
                        "t": float(mk.get("tag_size_mm", cfg.get("tag_size_mm", 0)))})
    tags = sorted({round(m["t"], 2) for m in markers})
    ids = [m["id"] for m in markers]
    tgt = cfg.get("target", {})
    tlo, thi = V.min(0), V.max(0)
    return {
        "name": label or model_dir.name,
        "parts": parts,
        "lo": np.round(lo, 3).tolist(), "hi": np.round(hi, 3).tolist(),
        "markers": markers,
        "info": {
            "size": f"{thi[0] - tlo[0]:.1f} x {thi[1] - tlo[1]:.1f} x {thi[2] - tlo[2]:.1f} mm",
            "markers": len(markers),
            "ids": f"{min(ids)}-{max(ids)}" if ids else "-",
            "tags": " / ".join(f"{t:g}" for t in tags) + " mm" if tags else "-",
            "dict": cfg.get("dict", "-"),
            "body": tgt.get("body") or ("white" if white.mean() > 0.5 else "black"),
            "margin": tgt.get("margin_mm", (cfg.get("markers") or [{}])[0].get("margin_mm")),
            "fixture": fixture_name if fixture is not None else None,
        },
    }


def write_viewer(scenes, out_html, title: str = "aprilcylinder viewer", standalone: bool = True) -> Path:
    """scenes: list of scene_payload() dicts or model folders.

    standalone=False leaves out the <!doctype>/<head> wrapper (for hosts that add their own).
    """
    payload = [s if isinstance(s, dict) else scene_payload(s) for s in scenes]
    html = _TEMPLATE.replace("__TITLE__", title).replace("__THREE__", THREE)
    html = html.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    if standalone:
        html = '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n' + html
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def export_obj(model_dir, out_obj) -> Path:
    """Wavefront OBJ + MTL with two materials (marker_black, body_white), units mm.

    Opens in Blender, MeshLab, Fusion 360 (mesh) and SolidWorks (as a graphics body).
    """
    V, T = load_painted_mesh(model_dir)
    out = Path(out_obj)
    out.parent.mkdir(parents=True, exist_ok=True)
    mtl = out.with_suffix(".mtl")
    mtl.write_text("newmtl black\nKd 0.06 0.06 0.06\nKa 0 0 0\nKs 0.05 0.05 0.05\n\n"
                   "newmtl white\nKd 0.95 0.95 0.93\nKa 0 0 0\nKs 0.05 0.05 0.05\n", encoding="utf-8")
    lines = [f"# aprilcylinder export of {Path(model_dir).name}, units mm, +Z = target axis\n", f"mtllib {mtl.name}\n"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in V]
    for color in ("white", "black"):
        lines.append(f"g {color}\nusemtl {color}\n")
        lines += [f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c, w in T if (color == "white") == w]
    out.write_text("".join(lines), encoding="utf-8")
    return out


_TEMPLATE = r"""<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: one-screen viewer. The 3D view fills the page; a slim instrument bar on top, a spec card bottom-left. */
:root {
  --bg: #e9ecef;          /* bench grey, slightly cool */
  --panel: rgba(250, 251, 252, 0.88);
  --ink: #1c2126;
  --muted: #5d6670;
  --line: #c9cfd6;
  --accent: #1f6feb;      /* calibration blue */
  --tag: #d9480f;         /* marker ID labels */
  --tag-bg: #fff7f2;
  --grid: #c3c9d0;
  --fixture: #8f9aa6;     /* fixed part (printed fixture) */
  --font-ui: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #15191e; --panel: rgba(28, 33, 39, 0.9); --ink: #e6e9ec; --muted: #9aa4ae; --line: #36404a;
  --accent: #6ea8ff; --tag: #ff9d6c; --tag-bg: #2a1d16; --grid: #2c343c; --fixture: #6b7783; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #15191e; --panel: rgba(28, 33, 39, 0.9); --ink: #e6e9ec; --muted: #9aa4ae; --line: #36404a;
  --accent: #6ea8ff; --tag: #ff9d6c; --tag-bg: #2a1d16; --grid: #2c343c; --fixture: #6b7783; color-scheme: dark; }
html, body { height: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--font-ui); overflow: hidden; }
#stage { position: fixed; inset: 0; }
#stage canvas { display: block; width: 100%; height: 100%; touch-action: none; }
#labels { position: fixed; inset: 0; pointer-events: none; overflow: hidden; }
.lbl { position: absolute; transform: translate(-50%, -50%); font: 500 11px/1 var(--font-data); color: var(--tag);
  background: var(--tag-bg); border: 1px solid var(--tag); border-radius: 3px; padding: 2px 4px; white-space: nowrap; }
.bar { position: fixed; top: env(safe-area-inset-top, 0px); left: 0; right: 0; display: flex; flex-wrap: wrap; gap: 8px 14px;
  align-items: center; padding-block: 10px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  background: var(--panel); border-bottom: 1px solid var(--line); backdrop-filter: blur(6px); z-index: 2; }
.brand { font-weight: 600; letter-spacing: 0.01em; margin-right: 4px; }
.brand span { color: var(--muted); font-weight: 400; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; min-width: 0; }
button, .toggle { font: 500 13px/1 var(--font-ui); color: var(--ink); background: transparent; border: 1px solid var(--line);
  border-radius: 999px; padding: 7px 12px; cursor: pointer; }
button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tools { display: flex; flex-wrap: wrap; gap: 6px; margin-left: auto; }
.card { position: fixed; left: max(16px, env(safe-area-inset-left, 0px)); bottom: calc(16px + env(safe-area-inset-bottom, 0px));
  width: min(340px, calc(100% - 32px)); background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px 14px; display: grid; gap: 10px; z-index: 2; backdrop-filter: blur(6px); }
.card h2 { margin: 0; font-size: 15px; font-weight: 600; text-wrap: balance; overflow-wrap: anywhere; }
.spec { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; margin: 0; font-size: 12.5px; }
.spec dt { color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-size: 10.5px; padding-top: 2px; }
.spec dd { margin: 0; font-family: var(--font-data); font-variant-numeric: tabular-nums; min-width: 0; overflow-wrap: anywhere; }
.dial { display: grid; gap: 6px; }
.dial label { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); }
.dial output { font-family: var(--font-data); color: var(--ink); font-variant-numeric: tabular-nums; }
input[type=range] { width: 100%; accent-color: var(--accent); }
.seen { font: 12px/1.4 var(--font-data); color: var(--muted); min-height: 1.4em; overflow-wrap: anywhere; }
.seen b { color: var(--tag); font-weight: 500; }
.hint { position: fixed; right: max(16px, env(safe-area-inset-right, 0px)); bottom: calc(16px + env(safe-area-inset-bottom, 0px));
  font-size: 12px; color: var(--muted); z-index: 2; text-align: right; }
.err { position: fixed; inset: 0; display: grid; place-items: center; padding-inline: 16px; color: var(--muted); text-align: center; }
@media (max-width: 640px) {
  .tools { margin-left: 0; }
  .card { width: calc(100% - 32px); }
  .hint { display: none; }
  .spec { grid-template-columns: auto 1fr auto 1fr; }
}
</style>

<div id="stage" aria-label="3D view of the target"></div>
<div id="labels"></div>
<header class="bar">
  <div class="brand">aprilcylinder <span>3D</span></div>
  <div class="chips" id="models" role="group" aria-label="Model"></div>
  <div class="tools" role="group" aria-label="View options">
    <button id="t-spin" aria-pressed="true">Spin</button>
    <button id="t-ids" aria-pressed="true">Tag IDs</button>
    <button id="t-axes" aria-pressed="true">Axes</button>
    <button id="t-reset">Reset view</button>
  </div>
</header>
<section class="card" aria-live="polite">
  <h2 id="m-name">Loading model</h2>
  <dl class="spec">
    <dt>Size</dt><dd id="m-size">-</dd>
    <dt>Markers</dt><dd id="m-markers">-</dd>
    <dt>Tag</dt><dd id="m-tags">-</dd>
    <dt>Dict</dt><dd id="m-dict">-</dd>
    <dt>Joint</dt><dd id="m-joint">-</dd>
  </dl>
  <div class="dial">
    <label for="angle">Joint angle, rotation about Z <output id="angle-out">0.0&deg;</output></label>
    <input id="angle" type="range" min="0" max="360" step="0.5" value="0">
  </div>
  <div class="seen" id="seen">Tags facing the camera: -</div>
</section>
<div class="hint">Drag to orbit &middot; scroll to zoom &middot; right-drag to pan</div>

<script type="importmap">
{ "imports": { "three": "__THREE__/build/three.module.js", "three/addons/": "__THREE__/examples/jsm/" } }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MODELS = __DATA__;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const b64 = (s, T) => { const b = atob(s); const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };

const stage = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
stage.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 1, 5000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f96, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.5); sun.position.set(300, -400, 600); scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(-400, 300, 200); scene.add(fill);

const pivot = new THREE.Group(); scene.add(pivot);       // rotates about the target axis (Z)
let grid = null, axes = null, current = null;
const labelsEl = document.getElementById("labels");
let labelEls = [];
const state = { spin: !reduceMotion, ids: true, axes: true, angle: 0 };

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", setTheme);
new MutationObserver(setTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

let meshes = [];
let fixtureMats = [];
function setTheme() {
  scene.background = new THREE.Color(css("--bg"));
  if (grid) { grid.material.color = new THREE.Color(css("--grid")); }
  fixtureMats.forEach((mat) => mat.color = new THREE.Color(css("--fixture")));
}
function partMesh(p) {
  const V = b64(p.v, Float32Array), I = b64(p.i, p.i32 ? Uint32Array : Uint16Array);
  const W = p.w ? b64(p.w, Uint8Array) : null;
  const n = I.length; const pos = new Float32Array(n * 3); const col = W ? new Float32Array(n * 3) : null;
  const white = [0.93, 0.93, 0.91], black = [0.07, 0.07, 0.08];
  for (let t = 0; t < n / 3; t++) {
    for (let k = 0; k < 3; k++) {
      const vi = I[t * 3 + k], o = (t * 3 + k) * 3;
      pos[o] = V[vi * 3]; pos[o + 1] = V[vi * 3 + 1]; pos[o + 2] = V[vi * 3 + 2];
      if (col) col.set(W[t] ? white : black, o);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  if (col) g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({ vertexColors: !!col, flatShading: true, roughness: col ? 0.75 : 0.6, metalness: 0 });
  if (!col) { mat.color = new THREE.Color(css("--fixture")); fixtureMats.push(mat); }
  return new THREE.Mesh(g, mat);
}

function build(m) {
  meshes.forEach((x) => { x.parent.remove(x); x.geometry.dispose(); x.material.dispose(); });
  meshes = []; fixtureMats = [];
  if (axes) pivot.remove(axes);
  if (grid) scene.remove(grid);
  m.parts.forEach((p) => { const me = partMesh(p); (p.joint === "revolute" ? pivot : scene).add(me); meshes.push(me); });
  const size = Math.max(m.hi[0] - m.lo[0], m.hi[1] - m.lo[1], m.hi[2] - m.lo[2]);
  axes = new THREE.AxesHelper(size * 0.55); axes.visible = state.axes; pivot.add(axes);
  const span = Math.ceil(size * 2.4 / 10) * 10;
  grid = new THREE.GridHelper(span, span / 10); grid.rotation.x = Math.PI / 2; grid.position.z = m.lo[2] - 0.05;
  grid.material.transparent = true; grid.material.opacity = 0.7; scene.add(grid);
  labelsEl.replaceChildren(); labelEls = m.markers.map((mk) => {
    const d = document.createElement("div"); d.className = "lbl"; d.textContent = mk.id; labelsEl.appendChild(d); return d;
  });
  current = m; resetView(); setTheme(); showInfo(m);
}

function resetView() {
  const m = current; const c = new THREE.Vector3((m.lo[0] + m.hi[0]) / 2, (m.lo[1] + m.hi[1]) / 2, (m.lo[2] + m.hi[2]) / 2);
  const r = Math.hypot(m.hi[0] - m.lo[0], m.hi[1] - m.lo[1], m.hi[2] - m.lo[2]) / 2;
  const d = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.6;
  const el = THREE.MathUtils.degToRad(22), az = THREE.MathUtils.degToRad(-60);
  camera.position.set(c.x + d * Math.cos(el) * Math.cos(az), c.y + d * Math.cos(el) * Math.sin(az), c.z + d * Math.sin(el));
  controls.target.copy(c); controls.update();
}

function showInfo(m) {
  document.getElementById("m-name").textContent = m.name;
  document.getElementById("m-size").textContent = m.info.size;
  document.getElementById("m-markers").textContent = `${m.info.markers} (IDs ${m.info.ids})`;
  document.getElementById("m-tags").textContent = m.info.tags + (m.info.margin != null ? `, margin ${m.info.margin} mm` : "");
  document.getElementById("m-dict").textContent = `${m.info.dict}, ${m.info.body} body`;
  document.getElementById("m-joint").textContent = m.info.fixture ? `1-DoF revolute about Z, on ${m.info.fixture}` : "1-DoF revolute about Z";
}

const chips = document.getElementById("models");
MODELS.forEach((m, k) => {
  const b = document.createElement("button"); b.id = `model-${k}`; b.textContent = m.name;
  b.setAttribute("aria-pressed", k === 0 ? "true" : "false");
  b.addEventListener("click", () => { chips.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); build(m); });
  chips.appendChild(b);
});
if (MODELS.length < 2) chips.hidden = true;

function toggle(id, key, after) {
  const b = document.getElementById(id); b.setAttribute("aria-pressed", String(state[key]));
  b.addEventListener("click", () => { state[key] = !state[key]; b.setAttribute("aria-pressed", String(state[key])); after && after(); });
}
toggle("t-spin", "spin"); toggle("t-ids", "ids"); toggle("t-axes", "axes", () => { axes.visible = state.axes; });
document.getElementById("t-reset").addEventListener("click", resetView);
const angleIn = document.getElementById("angle"), angleOut = document.getElementById("angle-out");
angleIn.addEventListener("input", () => { state.angle = parseFloat(angleIn.value); state.spin = false;
  document.getElementById("t-spin").setAttribute("aria-pressed", "false"); });

function resize() { const w = stage.clientWidth, h = stage.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
window.addEventListener("resize", resize);

const v = new THREE.Vector3(), nrm = new THREE.Vector3(), toCam = new THREE.Vector3();
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (state.spin) { state.angle = (state.angle + dt * 24) % 360; angleIn.value = state.angle.toFixed(1); }
  pivot.rotation.z = THREE.MathUtils.degToRad(state.angle);
  angleOut.textContent = `${state.angle.toFixed(1)}°`;
  controls.update(); renderer.render(scene, camera);
  const seen = []; const w = stage.clientWidth, h = stage.clientHeight;
  pivot.updateMatrixWorld();
  current.markers.forEach((mk, k) => {
    v.set(mk.c[0], mk.c[1], mk.c[2]).applyMatrix4(pivot.matrixWorld);
    nrm.set(mk.n[0], mk.n[1], mk.n[2]).transformDirection(pivot.matrixWorld);
    toCam.copy(camera.position).sub(v).normalize();
    const facing = nrm.dot(toCam);
    if (facing > 0.26) seen.push(mk.id);                       // within ~75 deg of the line of sight
    const el = labelEls[k];
    if (!state.ids || facing <= 0.05) { el.hidden = true; return; }
    v.project(camera); el.hidden = false;
    el.style.left = `${(v.x * 0.5 + 0.5) * w}px`; el.style.top = `${(-v.y * 0.5 + 0.5) * h}px`;
  });
  document.getElementById("seen").innerHTML = seen.length
    ? `Tags facing the camera: <b>${seen.join(", ")}</b>` : "Tags facing the camera: none";
  requestAnimationFrame(frame);
}

if (!MODELS.length) { document.body.insertAdjacentHTML("beforeend", '<p class="err">No model data in this page.</p>'); }
else { resize(); build(MODELS[0]); requestAnimationFrame(frame); }
</script>
"""
