"""Put markers on the flat square faces of an existing STL part (e.g. a band you modelled in CAD).

Every flat face that is exactly a square made of two triangles (at least ``min_size`` mm) gets one
marker. Only those faces are re-meshed; the rest of the part is kept as is, so its shape and volume
do not change. Edges of neighbouring triangles are split to match, keeping the mesh watertight.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mesh import (Mesh, load_stl, marker_bits, marker_pixels, max_marker_id, render_preview, reorient_up,
                   write_3mf)

UP = np.array([0.0, 0.0, 1.0])


def find_square_faces(tris: np.ndarray, min_size: float) -> list[dict]:
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    dist = (nrm * tris[:, 0]).sum(1)
    groups: dict = {}
    for i in range(len(tris)):
        groups.setdefault((tuple(np.round(nrm[i], 3) + 0.0), round(float(dist[i]), 2)), []).append(i)
    faces = []
    for idx in groups.values():
        if len(idx) != 2:
            continue
        pts = {Mesh.key(p): p for i in idx for p in tris[i]}
        if len(pts) != 4:
            continue
        p = np.array(list(pts.values()))
        c, n = p.mean(0), nrm[idx[0]]
        e1 = (p[0] - c) / np.linalg.norm(p[0] - c)
        e2 = np.cross(n, e1)
        p = p[np.argsort([np.arctan2((q - c) @ e2, (q - c) @ e1) for q in p])]
        sides = [np.linalg.norm(p[(k + 1) % 4] - p[k]) for k in range(4)]
        diag = [np.linalg.norm(p[2] - p[0]), np.linalg.norm(p[3] - p[1])]
        if max(sides) - min(sides) > 0.01 * max(sides) or abs(diag[0] - diag[1]) > 0.01 * diag[0]:
            continue
        if min(sides) < min_size:
            continue
        faces.append({"corners": p, "normal": n, "tris": set(idx), "side": float(np.mean(sides))})
    return faces


def paint_stl(stl_path, out_dir, *, up_axis="z", dictionary="4x4_1000", first_id=0, margin_mm=1.0,
              body="white", min_size=15.0, name=None, preview=True) -> dict:
    tris = reorient_up(load_stl(stl_path), up_axis)
    faces = find_square_faces(tris, min_size)
    if not faces:
        raise ValueError("no flat square faces found (each marker face must be a square of two triangles)")
    side = sorted((f for f in faces if abs(f["normal"] @ UP) < 0.5),
                  key=lambda f: np.arctan2(f["normal"][1], f["normal"][0]) % (2 * np.pi))
    caps = sorted((f for f in faces if abs(f["normal"] @ UP) >= 0.5), key=lambda f: -f["normal"][2])
    mp = marker_pixels(dictionary)
    body_white = body == "white"

    mesh, markers, edge_pts, replaced = Mesh(), [], {}, set()
    next_id = first_id
    for k, f in enumerate(side + caps):
        n, p = f["normal"], f["corners"]
        if abs(n @ UP) < 0.5:
            right, down, fname = np.cross(UP, n), -UP, f"side{k}"
        else:
            sign = 1.0 if n[2] > 0 else -1.0
            edges = [p[(i + 1) % 4] - p[i] for i in range(4)]
            right = max(edges, key=lambda e: e[0] * sign)
            right = right / np.linalg.norm(right)
            down, fname = np.cross(right, n), ("+Z" if sign > 0 else "-Z")
        right = right / np.linalg.norm(right)
        u, v = p @ right, p @ down
        tl, tr, bl = p[np.argmin(u + v)], p[np.argmax(u - v)], p[np.argmax(v - u)]
        br = p[np.argmax(u + v)]
        m = margin_mm / f["side"]
        if m * 2 >= 1:
            raise ValueError(f"margin {margin_mm} mm is too large for a {f['side']:.1f} mm face")
        frac = ([0.0] if m > 0 else []) + [m + (1 - 2 * m) * i / mp for i in range(mp + 1)] + ([1.0] if m > 0 else [])
        b = 1 if m > 0 else 0
        nc = len(frac) - 1
        grid = lambda r, c, frac=frac, tl=tl, tr=tr, bl=bl: tl + (tr - tl) * frac[c] + (bl - tl) * frac[r]  # noqa: E731
        bits = marker_bits(dictionary, next_id)
        for r in range(nc):
            for c in range(nc):
                inside = b <= r < b + mp and b <= c < b + mp
                white = bool(bits[r - b, c - b]) if inside else True
                mesh.quad(grid(r, c), grid(r, c + 1), grid(r + 1, c + 1), grid(r + 1, c), n,
                          painted=(white != body_white))
        for a, e, line in ((tl, tr, [grid(0, c) for c in range(1, nc)]), (bl, br, [grid(nc, c) for c in range(1, nc)]),
                           (tl, bl, [grid(r, 0) for r in range(1, nc)]), (tr, br, [grid(r, nc) for r in range(1, nc)])):
            edge_pts[frozenset((Mesh.key(a), Mesh.key(e)))] = (a, line)
        cell = float(np.linalg.norm(grid(b, b + 1) - grid(b, b)))
        quad = [grid(b, b), grid(b, b + mp), grid(b + mp, b + mp), grid(b + mp, b)]
        markers.append({"id": next_id, "face": fname, "normal": np.round(n, 6).tolist(),
                        "tag_size_mm": round(cell * mp, 5), "cell_size_mm": round(cell, 5),
                        "corners_mm": [np.round(q, 5).tolist() for q in quad]})
        replaced |= f["tris"]
        next_id += 1
    if next_id - 1 > max_marker_id(dictionary):
        raise ValueError(f"IDs exceed dictionary {dictionary}")

    def add_split(a, e, c, n):
        tri = (a, e, c)
        for i in range(3):
            p, q, r = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
            hit = edge_pts.get(frozenset((Mesh.key(p), Mesh.key(q))))
            if hit is None:
                continue
            start, line = hit
            chain = [p] + (line if Mesh.key(start) == Mesh.key(p) else line[::-1]) + [q]
            for s, t in zip(chain[:-1], chain[1:]):
                add_split(s, t, r, n)
            return
        mesh.tri(a, e, c, n, painted=not body_white)

    for i, t in enumerate(tris):
        if i not in replaced:
            add_split(t[0], t[1], t[2], np.cross(t[1] - t[0], t[2] - t[0]))

    vol0 = float(sum(t[0] @ np.cross(t[1], t[2]) for t in tris) / 6.0)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = name or Path(stl_path).stem
    write_3mf(out / f"{name}.3mf", mesh, name, body=body)
    lo, hi = mesh.bounds()
    ids = [mk["id"] for mk in markers]
    faces_map: dict[str, list[int]] = {}
    for mk in markers:
        faces_map.setdefault(mk["face"], []).append(mk["id"])
    config = {
        "schema_version": 1,
        "target": {"type": "painted_stl", "source": Path(stl_path).name, "up_axis_in_stl": up_axis,
                   "frame": "STL origin, +Z = the STL axis given by --up-axis", "body": body,
                   "margin_mm": margin_mm, "height_mm": round(float(hi[2] - lo[2]), 4)},
        "dict": dictionary, "grid": "1x1x1", "tag_ids": ids,
        "tag_size_mm": markers[0]["tag_size_mm"], "cell_size_mm": markers[0]["cell_size_mm"],
        "margin_cells": 1, "border_cells": 1, "marker_pixels": mp,
        "box_dims": [round(float(x), 4) for x in hi - lo],
        "faces": faces_map,
        "markers": markers,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    sizes = sorted({(round(f["side"], 2), mk["tag_size_mm"]) for f, mk in zip(side + caps, markers)})
    lines = [f"{Path(stl_path).name}: {len(side)} side + {len(caps)} cap markers, IDs {ids[0]}-{ids[-1]} ({dictionary})",
             "   ".join(f"face {s:g} mm -> tag {t:.2f} mm" for s, t in sizes) + f"   margin {margin_mm:g} mm, body {body}"]
    if preview:
        render_preview(out / "thumbnail.png", mesh, config, lines, body=body)
    for line in lines:
        print(line)
    print(f"mesh closed: {mesh.is_closed()}, volume {mesh.volume():.1f} mm3 (original {vol0:.1f}), "
          f"triangles {len(mesh.tris)}")
    print(f"wrote {out / (name + '.3mf')}, config.json" + (", thumbnail.png" if preview else ""))
    return config
