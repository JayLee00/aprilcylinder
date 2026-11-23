"""Build the README animation (docs/demo.gif) and the GitHub Pages viewer (docs/index.html).

    pip install -e . pillow
    python docs/make_media.py
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aprilcylinder.cylinder import generate, load_spec  # noqa: E402
from aprilcylinder.synth import load_painted_mesh  # noqa: E402
from aprilcylinder.viewer import demo_fixture, scene_payload, write_viewer  # noqa: E402

OUT = ROOT / "out" / "media"
DOCS = ROOT / "docs"


def rot(ax, deg):
    v = np.zeros(3)
    v[ax] = math.radians(deg)
    return cv2.Rodrigues(v)[0]


def render_frame(parts, markers, angle, size=(560, 600), seat=0.0):
    """parts: list of (V, F, colours per triangle as BGR tuples, revolute: bool)."""
    W, H = size
    f = 1450.0
    K = np.array([[f, 0, W / 2], [0, f, H * 0.53], [0, 0, 1.0]])
    Rc = rot(0, 90 + 24) @ rot(2, -35)          # camera looking slightly down at the assembly
    centre = np.array([0.0, 0.0, 78.0])
    t = np.array([0.0, 0.0, 600.0]) - Rc @ centre
    Rj = rot(2, angle)
    light = np.array([0.35, -0.55, -0.75])
    light /= np.linalg.norm(light)
    img = np.full((H, W, 3), (244, 241, 238), np.uint8)
    tris = []
    for V, F, cols, revolute in parts:
        P = (V @ Rj.T) if revolute else V
        Pc = P @ Rc.T + t
        uv = (Pc @ K.T)
        uv = uv[:, :2] / uv[:, 2:3]
        for k, (a, b, c) in enumerate(F):
            n = np.cross(Pc[b] - Pc[a], Pc[c] - Pc[a])
            if n @ Pc[a] >= 0:
                continue
            shade = 0.72 + 0.28 * max(0.0, float(n @ light) / (np.linalg.norm(n) + 1e-12))
            col = tuple(int(x * shade) for x in cols[k])
            tris.append((Pc[a][2] + Pc[b][2] + Pc[c][2], uv[[a, b, c]], col))
    for _, pts, col in sorted(tris, key=lambda q: -q[0]):
        cv2.fillConvexPoly(img, np.round(pts * 16).astype(np.int32), col, cv2.LINE_AA, 4)
    # joint axis
    top = max(float(V[:, 2].max()) for V, _, _, rev in parts if rev)
    ax = np.array([[0, 0, top + 3.0], [0, 0, top + 42.0]]) @ Rc.T + t
    ax = (ax @ K.T)
    ax = ax[:, :2] / ax[:, 2:3]
    p0, p1 = tuple(int(x) for x in ax[0]), tuple(int(x) for x in ax[1])
    cv2.arrowedLine(img, p0, p1, (205, 110, 30), 2, cv2.LINE_AA, tipLength=0.22)
    cv2.putText(img, "Z", (p1[0] + 8, p1[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (205, 110, 30), 2, cv2.LINE_AA)
    # IDs of markers facing the camera
    cam = -Rc.T @ t
    seen = []
    for mk in markers:
        c = Rj @ np.asarray(mk["c"])
        n = Rj @ np.asarray(mk["n"])
        if n @ ((cam - c) / np.linalg.norm(cam - c)) < 0.26:
            continue
        seen.append(mk["id"])
        p = K @ (Rc @ c + t)
        x, y = int(p[0] / p[2]), int(p[1] / p[2])
        txt = str(mk["id"])
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(img, (x - tw // 2 - 4, y - th // 2 - 4), (x + tw // 2 + 4, y + th // 2 + 4), (242, 247, 255), -1)
        cv2.rectangle(img, (x - tw // 2 - 4, y - th // 2 - 4), (x + tw // 2 + 4, y + th // 2 + 4), (15, 72, 217), 1)
        cv2.putText(img, txt, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (15, 72, 217), 1, cv2.LINE_AA)
    cv2.putText(img, f"joint angle about Z  {angle:5.1f} deg", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (40, 36, 33), 1, cv2.LINE_AA)
    cv2.putText(img, "tags in view: " + ", ".join(map(str, sorted(seen))), (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (90, 86, 82), 1, cv2.LINE_AA)
    return img


def main():
    from PIL import Image

    spec, _ = load_spec(ROOT / "examples" / "band8_d50_h120.yaml")
    generate(spec, OUT / "band8_d50_h120", preview=False)
    spec2, _ = load_spec(ROOT / "examples" / "two_bands_hex.yaml")
    generate(spec2, OUT / "two_bands_hex", preview=False)
    (fV, fF), seat = demo_fixture()

    scene = scene_payload(OUT / "band8_d50_h120", label="octagon band on a demo base", fixture=(fV, fF),
                          seat_z=seat, fixture_name="demo base")
    write_viewer([scene,
                  scene_payload(OUT / "band8_d50_h120", label="octagon band"),
                  scene_payload(OUT / "two_bands_hex", label="two hexagon bands")],
                 DOCS / "index.html", title="aprilcylinder viewer")

    V, T = load_painted_mesh(OUT / "band8_d50_h120")
    V = V + np.array([0, 0, seat])
    white, black, base = (236, 236, 232), (22, 21, 20), (166, 154, 143)
    parts = [(np.asarray(fV, float), fF, [base] * len(fF), False),
             (V, np.array([t[:3] for t in T]), [white if t[3] else black for t in T], True)]
    markers = [m for m in scene["markers"]]
    frames = [render_frame(parts, markers, a) for a in np.arange(0, 360, 7.5)]
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=96, method=Image.Quantize.MEDIANCUT)
            for f in frames]
    imgs[0].save(DOCS / "demo.gif", save_all=True, append_images=imgs[1:], duration=70, loop=0, optimize=True)
    print("wrote", DOCS / "demo.gif", "and", DOCS / "index.html")


if __name__ == "__main__":
    main()
