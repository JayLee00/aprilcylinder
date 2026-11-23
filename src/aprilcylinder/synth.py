"""Synthetic rendering of a generated target, for checking a design before printing.

Renders the painted 3MF mesh with flat shading, a little blur and noise, then runs the real
detector on it. It is optimistic compared with a camera (no motion blur, glare or colour bleed),
so treat the numbers as an upper bound on what a print can do.
"""
from __future__ import annotations

import glob
import json
import math
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .pose import CylinderTracker


def load_painted_mesh(model_dir) -> tuple[np.ndarray, list[tuple[int, int, int, bool]]]:
    """Vertices and triangles (a, b, c, is_white) from the first 3MF in a model folder."""
    files = sorted(glob.glob(str(Path(model_dir) / "*.3mf")))
    if not files:
        raise FileNotFoundError(f"no .3mf in {model_dir}")
    with zipfile.ZipFile(files[0]) as zf:
        name = next(n for n in zf.namelist() if n.startswith("3D/Objects/") and n.endswith(".model"))
        xml = zf.read(name).decode("utf-8")
        settings = zf.read("Metadata/model_settings.config").decode("utf-8")
    m = re.search(r'key="extruder" value="(\d+)"', settings)
    body_white = bool(m) and m.group(1) == "2"
    V = np.array([[float(a) for a in v] for v in re.findall(r'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"', xml)])
    T = []
    for a, b, c, p in re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"(?: paint_color="(\w+)")?', xml):
        painted = p != ""
        T.append((int(a), int(b), int(c), painted != body_white))   # white if (painted xor body white)
    return V, T


def render(V, T, K, R, t, size=(1280, 720), background=128, noise=3.0, seed=0) -> np.ndarray:
    rvec = cv2.Rodrigues(np.asarray(R, float))[0]
    img = np.full((size[1], size[0], 3), background, np.uint8)
    pv = cv2.projectPoints(V, rvec, np.asarray(t, float), K, None)[0].reshape(-1, 2)
    Vc = (R @ V.T).T + t
    for a, b, c, white in sorted(T, key=lambda q: -(Vc[q[0]][2] + Vc[q[1]][2] + Vc[q[2]][2])):
        n = np.cross(Vc[b] - Vc[a], Vc[c] - Vc[a])
        if n @ Vc[a] >= 0:
            continue
        shade = 0.8 + 0.2 * max(0.0, -n[2] / (np.linalg.norm(n) + 1e-12))
        g = int(235 * shade) if white else 25
        cv2.fillConvexPoly(img, np.round(pv[[a, b, c]] * 16).astype(np.int32), (g, g, g), cv2.LINE_AA, 4)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def rot(axis: int, deg: float) -> np.ndarray:
    v = np.zeros(3)
    v[axis] = math.radians(deg)
    return cv2.Rodrigues(v)[0]


VIEWS = {"side": 105.0, "oblique": 135.0, "top": 180.0}   # camera tilt about X: 90 = level, 180 = straight down


def selftest(model_dir, distance_mm=400.0, views=("side", "oblique", "top"), n=36, fx=900.0,
             size=(1280, 720), seed=1) -> dict:
    """Render `n` random angles per view and report detection rate and absolute angle error.

    Each frame uses a fresh tracker with no history, so the result is the single-image accuracy.
    """
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    h = float(cfg["target"].get("height_mm", cfg["box_dims"][2]))
    K = np.array([[fx, 0, size[0] / 2], [0, fx, size[1] / 2], [0, 0, 1.0]])
    V, T = load_painted_mesh(model_dir)
    rng = np.random.default_rng(seed)
    out = {}
    for view in views:
        base = rot(0, VIEWS[view])
        ok, errs, tags = 0, [], []
        for i, th in enumerate(rng.uniform(0, 360, n)):
            R = base @ rot(2, th)
            t = np.array([10.0, -5.0, distance_mm]) - R @ np.array([0, 0, h / 2])
            img = render(V, T, K, R, t, size=size, background=(128, 30, 225)[i % 3], seed=i)
            trk = CylinderTracker(model_dir, K, reference=base, enable_filter=False)
            st = trk.process(img)
            if not st.success:
                continue
            ok += 1
            tags.append(st.n_tags)
            errs.append(abs((st.angle_deg - th + 180.0) % 360.0 - 180.0))
        out[view] = {"detected": ok, "frames": n, "tags_per_frame": float(np.mean(tags)) if tags else 0.0,
                     "angle_err_mean_deg": float(np.mean(errs)) if errs else float("nan"),
                     "angle_err_max_deg": float(np.max(errs)) if errs else float("nan")}
    return out
