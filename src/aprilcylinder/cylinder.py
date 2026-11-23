"""Cylindrical fiducial targets: a cylinder with polygonal marker bands and marker caps.

A target is a stack of sections along +Z (bottom to top). Each section is either

* ``polygon`` - a regular N-sided prism ("band"); every side face can carry markers, or
* ``cylinder`` - a plain round section (tessellated with ``segments`` facets).

Neighbouring sections must be strictly nested in outline (one fully inside the other), which is
what a band around a round shaft looks like. The top and bottom faces can each carry one marker
in a centred square.

Frame of the generated model and of every pose reported by this package:
origin on the axis at the bottom face, +Z along the axis, +X through the centre of the first
polygon face (rotated by ``phase_deg``). Units: millimetres.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .mesh import Mesh, fan, marker_bits, marker_pixels, max_marker_id, render_preview, write_3mf, zipper

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Section:
    kind: str                      # "polygon" or "cylinder"
    height: float
    sides: int = 0
    apothem: float = 0.0           # polygon: centre to face
    radius: float = 0.0            # cylinder
    markers: bool = True           # polygon: put markers on the side faces
    rows: int = 1                  # polygon: markers stacked along Z on each face
    z0: float = 0.0
    z1: float = 0.0
    faces: list = field(default_factory=list)   # filled by the builder (polygon)

    @property
    def face_width(self) -> float:
        return 2.0 * self.apothem * math.tan(math.pi / self.sides)


@dataclass
class TargetSpec:
    sections: list[Section]
    dictionary: str = "4x4_1000"
    first_id: int = 0
    margin_mm: float = 1.0
    body: str = "white"            # "white" (white body, black marker cells) or "black"
    segments: int = 128
    phase_deg: float = 0.0
    top_marker: bool = True
    bottom_marker: bool = False
    top_square_mm: float | None = None
    bottom_square_mm: float | None = None
    name: str = "aprilcylinder"


# ---------------------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------------------
def _polygon_from(d: dict) -> Section:
    sides = int(d["sides"])
    if sides < 3:
        raise ValueError("polygon sides must be >= 3")
    if "apothem_mm" in d:
        apothem = float(d["apothem_mm"])
    elif "across_flats_mm" in d:
        apothem = float(d["across_flats_mm"]) / 2.0
    elif "face_mm" in d:
        apothem = float(d["face_mm"]) / (2.0 * math.tan(math.pi / sides))
    else:
        raise ValueError("polygon section needs face_mm, apothem_mm or across_flats_mm")
    width = 2.0 * apothem * math.tan(math.pi / sides)
    return Section("polygon", float(d.get("height_mm", width)), sides=sides, apothem=apothem,
                   markers=bool(d.get("markers", True)), rows=int(d.get("rows", 1)))


def spec_from_dict(d: dict) -> TargetSpec:
    sections = []
    for s in d["sections"]:
        kind = str(s.get("type", "cylinder")).lower()
        if kind in ("polygon", "band", "prism"):
            sections.append(_polygon_from(s))
        elif kind in ("cylinder", "round"):
            r = float(s["diameter_mm"]) / 2.0 if "diameter_mm" in s else float(s["radius_mm"])
            sections.append(Section("cylinder", float(s["height_mm"]), radius=r))
        else:
            raise ValueError(f"unknown section type '{kind}'")
    caps = d.get("caps", {}) or {}
    top, bottom = caps.get("top", {}) or {}, caps.get("bottom", {}) or {}
    return TargetSpec(
        sections=sections,
        dictionary=str(d.get("dictionary", "4x4_1000")),
        first_id=int(d.get("first_id", d.get("ids_start", 0))),
        margin_mm=float(d.get("margin_mm", 1.0)),
        body=str(d.get("body", "white")),
        segments=int(d.get("segments", 128)),
        phase_deg=float(d.get("phase_deg", 0.0)),
        top_marker=bool(top.get("marker", True)),
        bottom_marker=bool(bottom.get("marker", False)),
        top_square_mm=top.get("square_mm"),
        bottom_square_mm=bottom.get("square_mm"),
        name=str(d.get("name", Path(str(d.get("output", "aprilcylinder"))).name)),
    )


def load_spec(path) -> tuple[TargetSpec, str | None]:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return spec_from_dict(d), d.get("output")


def simple_spec(diameter: float, height: float, sides: int = 8, face: float | None = None,
                band_height: float | None = None, bands: str = "bottom", **kw) -> TargetSpec:
    """Cylinder of `diameter` x `height` with an N-sided marker band at the bottom, top or both.

    face: band face width (default: the smallest face whose band clears the cylinder by 1.5 mm).
    band_height: default = face width (square faces). Band height counts toward `height`.
    """
    r = diameter / 2.0
    apothem = (face / (2.0 * math.tan(math.pi / sides))) if face else r + 1.5
    if apothem <= r:
        raise ValueError(f"band face {face} mm is too small: its apothem {apothem:.2f} mm must exceed the "
                         f"cylinder radius {r:g} mm")
    width = 2.0 * apothem * math.tan(math.pi / sides)
    bh = band_height or width
    n_bands = {"bottom": 1, "top": 1, "both": 2, "none": 0}[bands]
    rest = height - n_bands * bh
    if rest <= 0:
        raise ValueError("height is too small for the requested bands")
    band = lambda: Section("polygon", bh, sides=sides, apothem=apothem)  # noqa: E731
    secs = []
    if bands in ("bottom", "both"):
        secs.append(band())
    secs.append(Section("cylinder", rest, radius=r))
    if bands in ("top", "both"):
        secs.append(band())
    return TargetSpec(sections=secs, **kw)


# ---------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------
def _dedupe(vals: list[float], tol: float = 1e-6) -> list[float]:
    out: list[float] = []
    for v in sorted(vals):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _radii(sec: Section, segments: int) -> tuple[float, float]:
    """(min, max) distance of the section outline from the axis."""
    if sec.kind == "polygon":
        return sec.apothem, sec.apothem / math.cos(math.pi / sec.sides)
    return sec.radius * math.cos(math.pi / segments), sec.radius


class _Grid:
    """A flat rectangular face split into cells; markers are placed in some of the cells."""

    def __init__(self, origin, right, down, u_lines, v_lines, markers):
        self.origin, self.right, self.down = origin, right, down
        self.u, self.v = u_lines, v_lines
        self.markers = markers     # list of dicts with u0, v0, cell, bits

    def point(self, u: float, v: float) -> np.ndarray:
        return self.origin + self.right * u + self.down * v

    def is_white(self, uc: float, vc: float) -> bool:
        for m in self.markers:
            n = m["bits"].shape[0]
            if m["u0"] <= uc <= m["u0"] + n * m["cell"] and m["v0"] <= vc <= m["v0"] + n * m["cell"]:
                c = min(n - 1, int((uc - m["u0"]) / m["cell"]))
                r = min(n - 1, int((vc - m["v0"]) / m["cell"]))
                return bool(m["bits"][r, c])
        return True

    def emit(self, mesh: Mesh, normal, body_white: bool) -> None:
        for j in range(len(self.v) - 1):
            for i in range(len(self.u) - 1):
                u0, u1, v0, v1 = self.u[i], self.u[i + 1], self.v[j], self.v[j + 1]
                white = self.is_white((u0 + u1) / 2, (v0 + v1) / 2)
                mesh.quad(self.point(u0, v0), self.point(u1, v0), self.point(u1, v1), self.point(u0, v1),
                          normal, painted=(white != body_white))

    def boundary(self) -> list[np.ndarray]:
        U, V = self.u, self.v
        pts = [self.point(u, V[0]) for u in U] + [self.point(u, V[-1]) for u in U]
        pts += [self.point(U[0], v) for v in V[1:-1]] + [self.point(U[-1], v) for v in V[1:-1]]
        return pts

    def marker_corners(self, m) -> list[list[float]]:
        n = m["bits"].shape[0]
        t = n * m["cell"]
        q = [self.point(m["u0"], m["v0"]), self.point(m["u0"] + t, m["v0"]),
             self.point(m["u0"] + t, m["v0"] + t), self.point(m["u0"], m["v0"] + t)]
        return [np.round(p, 5).tolist() for p in q]   # TL, TR, BR, BL as seen from outside


def build(spec: TargetSpec, allow_no_markers: bool = False) -> tuple[Mesh, dict, list[str]]:
    """Build the mesh and the detector config. Returns (mesh, config, report lines)."""
    if spec.body not in ("white", "black"):
        raise ValueError("body must be 'white' or 'black'")
    mp = marker_pixels(spec.dictionary)
    m = spec.margin_mm
    if m < 0:
        raise ValueError("margin_mm must be >= 0")
    body_white = spec.body == "white"
    phase = math.radians(spec.phase_deg)
    next_id = spec.first_id
    markers: list[dict] = []
    report: list[str] = []

    def take_id() -> int:
        nonlocal next_id
        i = next_id
        next_id += 1
        return i

    z = 0.0
    for s in spec.sections:
        if s.height <= 0:
            raise ValueError("section heights must be positive")
        s.z0, s.z1 = z, z + s.height
        z = s.z1
    height = z

    for a, b in zip(spec.sections[:-1], spec.sections[1:]):
        a_min, a_max = _radii(a, spec.segments)
        b_min, b_max = _radii(b, spec.segments)
        if not (a_min > b_max + 0.05 or b_min > a_max + 0.05):
            raise ValueError(f"sections at z={a.z1:g} mm overlap in outline: one must lie fully inside the other "
                             f"(radii {a_min:.2f}-{a_max:.2f} vs {b_min:.2f}-{b_max:.2f} mm)")

    mesh = Mesh()
    loops: list[tuple[list, list]] = []   # (bottom loop, top loop) per section

    for si, s in enumerate(spec.sections):
        if s.kind == "polygon":
            w, h = s.face_width, s.height
            if s.markers:
                t = min(w - 2 * m, h / s.rows - 2 * m)
                if t <= 0:
                    raise ValueError(f"section {si}: no room for markers (face {w:.2f} x {h:.2f} mm, margin {m} mm)")
                cell = t / mp
                u0 = (w - t) / 2
                u_lines = _dedupe([0.0, w] + [u0 + k * cell for k in range(mp + 1)])
                v_lines = [0.0, h]
                rows = []
                for r in range(s.rows):
                    v0 = r * h / s.rows + (h / s.rows - t) / 2
                    rows.append(v0)
                    v_lines += [v0 + k * cell for k in range(mp + 1)]
                v_lines = _dedupe(v_lines)
                report.append(f"band {si}: {s.sides} faces {w:.2f} x {h:.2f} mm, tag {t:.2f} mm, cell {cell:.3f} mm")
            else:
                u_lines, v_lines, rows, cell = [0.0, w], [0.0, h], [], 0.0
            bottom, top = [], []
            for k in range(s.sides):
                phi = phase + 2 * math.pi * k / s.sides
                n = np.array([math.cos(phi), math.sin(phi), 0.0])
                right = np.cross(UP, n)
                origin = n * s.apothem - right * (w / 2) + UP * s.z1
                face_markers = []
                for v0 in rows:
                    tag_id = take_id()
                    face_markers.append({"id": tag_id, "u0": u0, "v0": v0, "cell": cell,
                                         "bits": marker_bits(spec.dictionary, tag_id)})
                g = _Grid(origin, right, -UP, u_lines, v_lines, face_markers)
                g.emit(mesh, n, body_white)
                for fm in face_markers:
                    markers.append({"id": fm["id"], "face": f"s{si}f{k}", "normal": np.round(n, 6).tolist(),
                                    "tag_size_mm": round(mp * cell, 5), "cell_size_mm": round(cell, 5),
                                    "corners_mm": g.marker_corners(fm)})
                top += [g.point(u, 0.0) for u in u_lines[:-1]]
                bottom += [g.point(u, h) for u in u_lines[:-1]]
            loops.append((bottom, top))
        else:
            n_seg = spec.segments
            pts = [np.array([s.radius * math.cos(phase + 2 * math.pi * i / n_seg),
                             s.radius * math.sin(phase + 2 * math.pi * i / n_seg), 0.0]) for i in range(n_seg)]
            bottom = [p + UP * s.z0 for p in pts]
            top = [p + UP * s.z1 for p in pts]
            for i in range(n_seg):
                j = (i + 1) % n_seg
                mid = (pts[i] + pts[j]) / 2
                mesh.quad(bottom[i], bottom[j], top[j], top[i], mid, painted=not body_white)
            loops.append((bottom, top))

    # horizontal ledges between sections (body colour)
    for si in range(len(spec.sections) - 1):
        a, b = spec.sections[si], spec.sections[si + 1]
        a_min, _ = _radii(a, spec.segments)
        _, b_max = _radii(b, spec.segments)
        if a_min > b_max:      # lower section is wider: ledge faces up
            zipper(mesh, loops[si + 1][0], loops[si][1], UP, painted=not body_white)
        else:                  # upper section is wider: ledge faces down
            zipper(mesh, loops[si][1], loops[si + 1][0], -UP, painted=not body_white)

    # caps
    def cap(which: str):
        top_cap = which == "top"
        sec = spec.sections[-1] if top_cap else spec.sections[0]
        loop = loops[-1][1] if top_cap else loops[0][0]
        zc = height if top_cap else 0.0
        n = UP if top_cap else -UP
        want = spec.top_marker if top_cap else spec.bottom_marker
        square = spec.top_square_mm if top_cap else spec.bottom_square_mm
        if not want:
            fan(mesh, loop, UP * zc, n, painted=not body_white)
            return
        rmin, _ = _radii(sec, spec.segments)
        limit = (rmin - 0.3) * math.sqrt(2.0)
        S = float(square) if square else math.floor(limit * 100) / 100
        if S > limit:
            raise ValueError(f"{which} square {S:g} mm does not fit inside the {which} face (max {limit:.2f} mm)")
        t = S - 2 * m
        if t <= 0:
            raise ValueError(f"{which} square {S:g} mm leaves no room for a marker with margin {m} mm")
        cell = t / mp
        right = np.array([math.cos(phase), math.sin(phase), 0.0]) * (1.0 if top_cap else -1.0)
        down = np.cross(right, n)
        origin = UP * zc - right * (S / 2) - down * (S / 2)
        lines = _dedupe([0.0, S] + [m + k * cell for k in range(mp + 1)])
        tag_id = take_id()
        fm = {"id": tag_id, "u0": m, "v0": m, "cell": cell, "bits": marker_bits(spec.dictionary, tag_id)}
        g = _Grid(origin, right, down, lines, lines, [fm])
        g.emit(mesh, n, body_white)
        zipper(mesh, g.boundary(), loop, n, painted=not body_white)
        markers.append({"id": tag_id, "face": "+Z" if top_cap else "-Z", "normal": n.tolist(),
                        "tag_size_mm": round(t, 5), "cell_size_mm": round(cell, 5),
                        "corners_mm": g.marker_corners(fm)})
        report.append(f"{which} cap: square {S:.2f} mm, tag {t:.2f} mm, cell {cell:.3f} mm")

    cap("top")
    cap("bottom")

    if not markers and not allow_no_markers:
        raise ValueError("the target has no markers")
    if next_id - 1 > max_marker_id(spec.dictionary):
        raise ValueError(f"IDs {spec.first_id}-{next_id - 1} exceed dictionary {spec.dictionary} "
                         f"(max ID {max_marker_id(spec.dictionary)})")

    rmax = max(_radii(s, spec.segments)[1] for s in spec.sections)
    ids = [mk["id"] for mk in markers]
    faces: dict[str, list[int]] = {}
    for mk in markers:
        faces.setdefault(mk["face"], []).append(mk["id"])
    config = {
        "schema_version": 1,
        "target": {
            "type": "aprilcylinder",
            "frame": "origin on the axis at the bottom face, +Z = axis, +X = centre of first band face",
            "height_mm": round(height, 4),
            "phase_deg": spec.phase_deg,
            "body": spec.body,
            "margin_mm": m,
            "sections": [{"type": s.kind, "z0_mm": round(s.z0, 4), "z1_mm": round(s.z1, 4),
                          **({"sides": s.sides, "apothem_mm": round(s.apothem, 4),
                              "face_mm": round(s.face_width, 4)} if s.kind == "polygon"
                             else {"radius_mm": s.radius})} for s in spec.sections],
        },
        # fields read by aprilcube's detector
        "dict": spec.dictionary, "grid": "1x1x1", "tag_ids": ids,
        "tag_size_mm": markers[0]["tag_size_mm"] if markers else 0.0,
        "cell_size_mm": markers[0]["cell_size_mm"] if markers else 0.0,
        "margin_cells": 1, "border_cells": 1, "marker_pixels": mp,
        "box_dims": [round(2 * rmax, 4), round(2 * rmax, 4), round(height, 4)],
        "faces": faces,
        "markers": markers,
    }
    id_text = f"IDs {ids[0]}-{ids[-1]}, " if ids else ""
    report.insert(0, f"{len(spec.sections)} sections, height {height:.2f} mm, max diameter {2 * rmax:.2f} mm, "
                     f"{len(ids)} markers ({id_text}{spec.dictionary})")
    return mesh, config, report


def analytic_volume(spec: TargetSpec) -> float:
    v = 0.0
    for s in spec.sections:
        if s.kind == "polygon":
            area = s.sides * s.apothem ** 2 * math.tan(math.pi / s.sides)
        else:
            area = 0.5 * spec.segments * s.radius ** 2 * math.sin(2 * math.pi / spec.segments)
        v += area * s.height
    return v


def generate(spec: TargetSpec, out_dir, preview: bool = True) -> dict:
    """Build and write <out>/<name>.3mf, config.json and thumbnail.png. Returns the config."""
    mesh, config, report = build(spec)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = spec.name or out.name
    write_3mf(out / f"{name}.3mf", mesh, name, body=spec.body)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if preview:
        render_preview(out / "thumbnail.png", mesh, config, report[:3], body=spec.body)
    for line in report:
        print(line)
    print(f"mesh closed: {mesh.is_closed()}, volume {mesh.volume():.1f} mm3 "
          f"(expected {analytic_volume(spec):.1f}), triangles {len(mesh.tris)}")
    print(f"wrote {out / (name + '.3mf')}, config.json" + (", thumbnail.png" if preview else ""))
    return config
