"""Combine several targets (and plain STL parts) into one Bambu Studio 3MF.

Every input becomes a separate object that can be moved, rotated or deleted on its own in Bambu
Studio. Geometry, paint and body colour are kept. Do not scale the objects in the slicer: the
detector trusts the dimensions in each target's config.json.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np

from .mesh import Mesh, load_stl, mesh_xml, reorient_up, write_3mf_objects


def _find_inputs(paths) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_file():
            out.append(p)
        elif list(p.glob("*.3mf")):
            out.extend(sorted(p.glob("*.3mf")))
        else:
            out.extend(sorted(p.glob("*/*.3mf")))
    if not out:
        raise FileNotFoundError("no .3mf or .stl inputs found")
    return out


def _read_3mf(path) -> tuple[str, np.ndarray, np.ndarray, int]:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.startswith("3D/Objects/") and n.endswith(".model")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected a single-object 3MF made by aprilcube/aprilcylinder, "
                             f"found {len(names)} objects; add such files in Bambu Studio instead")
        xml = zf.read(names[0]).decode("utf-8")
        settings = zf.read("Metadata/model_settings.config").decode("utf-8") \
            if "Metadata/model_settings.config" in zf.namelist() else ""
    mesh = xml[xml.index("<mesh>"): xml.index("</mesh>") + len("</mesh>")]
    pts = np.array([[float(a) for a in m] for m in re.findall(r'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"', mesh)])
    m = re.search(r'key="extruder" value="(\d+)"', settings)
    return mesh, pts.min(0), pts.max(0), int(m.group(1)) if m else 1


def _read_stl(path, up_axis) -> tuple[str, np.ndarray, np.ndarray]:
    tris = reorient_up(load_stl(path), up_axis)
    mesh = Mesh()
    for t in tris:
        mesh.tri(t[0], t[1], t[2], np.cross(t[1] - t[0], t[2] - t[0]))
    lo, hi = mesh.bounds()
    return mesh_xml(mesh, "4"), lo, hi


def _layout(sizes, plate_w, plate_d, gap):
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    rows, row, w = [], [], 0.0
    for i in order:
        if row and w + gap + sizes[i][0] > plate_w - 2 * gap:
            rows.append(row)
            row, w = [], 0.0
        w += (gap if row else 0.0) + sizes[i][0]
        row.append(i)
    rows.append(row)
    total = sum(max(sizes[i][1] for i in r) for r in rows) + gap * (len(rows) - 1)
    pos, y = {}, (plate_d - total) / 2.0
    for r in rows:
        depth = max(sizes[i][1] for i in r)
        x = (plate_w - (sum(sizes[i][0] for i in r) + gap * (len(r) - 1))) / 2.0
        for i in r:
            pos[i] = (x + sizes[i][0] / 2.0, y + depth / 2.0)
            x += sizes[i][0] + gap
        y += depth + gap
    return pos


def merge(inputs, output, *, plate=(256.0, 256.0), gap=10.0, stl_filament=2, stl_up_axis="z") -> list[dict]:
    items = []
    for f in _find_inputs(inputs):
        if f.suffix.lower() == ".stl":
            xml, lo, hi = _read_stl(f, stl_up_axis)
            ext = stl_filament
        else:
            xml, lo, hi, ext = _read_3mf(f)
        items.append({"name": f.stem, "mesh_xml": xml, "lo": lo, "hi": hi, "extruder": ext})
    pos = _layout([(it["hi"][0] - it["lo"][0], it["hi"][1] - it["lo"][1]) for it in items], plate[0], plate[1], gap)
    for k, it in enumerate(items):
        it["position"] = (pos[k][0] - (it["lo"][0] + it["hi"][0]) / 2.0,
                          pos[k][1] - (it["lo"][1] + it["hi"][1]) / 2.0, -float(it["lo"][2]))
    write_3mf_objects(output, items)
    print(f"wrote {output}: {len(items)} objects")
    for k, it in enumerate(items):
        print(f"  {it['name']:40s} at ({pos[k][0]:.0f}, {pos[k][1]:.0f}) mm, body filament {it['extruder']}")
    return items
