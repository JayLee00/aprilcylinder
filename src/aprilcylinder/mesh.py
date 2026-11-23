"""Triangle mesh, ArUco bit patterns, Bambu Studio 3MF writer and a preview renderer."""
from __future__ import annotations

import json
import math
import struct
import zipfile
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50, "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250, "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50, "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250, "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50, "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250, "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
NS_BBL = "http://schemas.bambulab.com/package/2021"
REL_TYPE = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
XML = '<?xml version="1.0" encoding="UTF-8"?>\n'
IDENTITY = "1 0 0 0 1 0 0 0 1"

# Bambu Studio paint codes: "4" = filament 1, "8" = filament 2.
# Convention used everywhere in this package: filament 1 = black, filament 2 = white.
PAINT_FILAMENT = {1: "4", 2: "8"}


def marker_pixels(dict_name: str) -> int:
    """Marker size in cells including the black border (4x4 -> 6, apriltag_36h11 -> 8)."""
    if dict_name.startswith("apriltag_36h11"):
        return 8
    return int(dict_name.split("x")[0]) + 2


def marker_bits(dict_name: str, tag_id: int) -> np.ndarray:
    """(mp, mp) bool array, True = white cell, row 0 = top of the marker."""
    mp = marker_pixels(dict_name)
    d = cv2.aruco.getPredefinedDictionary(DICTS[dict_name])
    return cv2.aruco.generateImageMarker(d, int(tag_id), mp) > 127


def max_marker_id(dict_name: str) -> int:
    d = cv2.aruco.getPredefinedDictionary(DICTS[dict_name])
    return d.bytesList.shape[0] - 1


class Mesh:
    """Indexed triangle mesh. Each triangle carries a flag: True = painted (the non-body colour)."""

    def __init__(self):
        self.verts: list[tuple[float, float, float]] = []
        self.tris: list[tuple[int, int, int, bool]] = []
        self._map: dict[tuple[int, int, int], int] = {}

    @staticmethod
    def key(p) -> tuple[int, int, int]:
        return tuple(int(round(float(x) * 10000)) for x in p)

    def v(self, p) -> int:
        k = self.key(p)
        if k not in self._map:
            self._map[k] = len(self.verts)
            self.verts.append(tuple(float(x) for x in p))
        return self._map[k]

    def tri(self, a, b, c, outward, painted: bool = False) -> None:
        """Add a triangle, flipped if needed so that its normal points along `outward`."""
        a, b, c = (np.asarray(p, float) for p in (a, b, c))
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-10:
            return
        if n @ np.asarray(outward, float) < 0:
            b, c = c, b
        ia, ib, ic = self.v(a), self.v(b), self.v(c)
        if len({ia, ib, ic}) == 3:
            self.tris.append((ia, ib, ic, bool(painted)))

    def quad(self, p00, p10, p11, p01, outward, painted: bool = False) -> None:
        self.tri(p00, p10, p11, outward, painted)
        self.tri(p00, p11, p01, outward, painted)

    # -- checks -------------------------------------------------------------------------
    def is_closed(self) -> bool:
        """Every directed edge appears once and its reverse appears once (closed, consistently oriented)."""
        edges: dict[tuple[int, int], int] = {}
        for a, b, c, _ in self.tris:
            for e in ((a, b), (b, c), (c, a)):
                edges[e] = edges.get(e, 0) + 1
        return all(n == 1 and edges.get((e[1], e[0]), 0) == 1 for e, n in edges.items())

    def volume(self) -> float:
        V = np.asarray(self.verts)
        return float(sum(V[a] @ np.cross(V[b], V[c]) for a, b, c, _ in self.tris) / 6.0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        V = np.asarray(self.verts)
        return V.min(0), V.max(0)


def zipper(mesh: Mesh, inner: list, outer: list, normal, painted: bool = False) -> None:
    """Triangulate the ring between two closed loops that are star-shaped around the Z axis.

    Both loops are given as 3D points at the same height; `inner` must lie strictly inside `outer`.
    """
    ang = lambda p: math.atan2(p[1], p[0]) % (2 * math.pi)  # noqa: E731
    inner = [np.asarray(p, float) for p in sorted(inner, key=ang)]
    outer = [np.asarray(p, float) for p in sorted(outer, key=ang)]
    ni, no = len(inner), len(outer)
    # start from the outer point closest in angle to inner[0]
    a0 = ang(inner[0])
    j0 = min(range(no), key=lambda k: abs((ang(outer[k]) - a0 + math.pi) % (2 * math.pi) - math.pi))
    outer = outer[j0:] + outer[:j0]

    def ccw(a, b, c) -> float:   # z of (b - a) x (c - a): > 0 for a correctly winding stitch triangle
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    i = j = 0
    while i < ni or j < no:
        a, b = inner[i % ni], outer[j % no]
        na, nb = inner[(i + 1) % ni], outer[(j + 1) % no]
        ok_inner = i < ni and ccw(a, b, na) > 1e-9      # advance along the inner loop
        ok_outer = j < no and ccw(a, b, nb) > 1e-9      # advance along the outer loop
        if ok_inner and ok_outer:
            take_inner = np.linalg.norm(na - b) <= np.linalg.norm(nb - a)   # shorter new diagonal
        elif ok_inner or ok_outer:
            take_inner = ok_inner
        else:
            take_inner = j >= no or (i < ni and ang(na) - ang(a) <= ang(nb) - ang(b))
        if take_inner:
            mesh.tri(a, b, na, normal, painted)
            i += 1
        else:
            mesh.tri(a, b, nb, normal, painted)
            j += 1


def fan(mesh: Mesh, loop: list, center, normal, painted: bool = False) -> None:
    ang = lambda p: math.atan2(p[1], p[0]) % (2 * math.pi)  # noqa: E731
    loop = sorted(loop, key=ang)
    for k in range(len(loop)):
        mesh.tri(center, loop[k], loop[(k + 1) % len(loop)], normal, painted)


# ---------------------------------------------------------------------------------------
# Bambu Studio 3MF
# ---------------------------------------------------------------------------------------
def _model_head() -> str:
    return (XML + f'<model unit="millimeter" xml:lang="en-US" xmlns="{NS_CORE}" xmlns:BambuStudio="{NS_BBL}"'
            + f' xmlns:p="{NS_PROD}" requiredextensions="p">\n'
            + '  <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
            + '  <metadata name="Application">BambuStudio-02.00.00.00</metadata>\n')


def mesh_xml(mesh: Mesh, paint_code: str) -> str:
    out = ["<mesh>\n        <vertices>\n"]
    out += [f'          <vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>\n' for x, y, z in mesh.verts]
    out.append("        </vertices>\n        <triangles>\n")
    for a, b, c, painted in mesh.tris:
        paint = f' paint_color="{paint_code}"' if painted else ""
        out.append(f'          <triangle v1="{a}" v2="{b}" v3="{c}"{paint}/>\n')
    out.append("        </triangles>\n      </mesh>")
    return "".join(out)


def write_3mf_objects(path, objects: list[dict]) -> None:
    """Write a Bambu Studio 3MF with one independent object per entry.

    Each entry: {"name", "mesh_xml" (a <mesh> element), "extruder" (body filament), "position" (x, y, z)}.
    """
    rel_open = XML + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
    objs, builds, settings, instances, assemble, rels = [], [], [], [], [], []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, ob in enumerate(objects):
            mesh_id, obj_id = 2 * k + 1, 2 * k + 2
            part = f"/3D/Objects/object_{k + 1}.model"
            zf.writestr(part[1:], _model_head() + "  <resources>\n"
                        + f'    <object id="{mesh_id}" p:UUID="{uuid4()}" type="model">\n      {ob["mesh_xml"]}\n'
                        + "    </object>\n  </resources>\n  <build/>\n</model>\n")
            rels.append(f'  <Relationship Target="{part}" Id="rel-{k + 1}" Type="{REL_TYPE}"/>\n')
            tx, ty, tz = ob["position"]
            tf = f"{IDENTITY} {tx:.4f} {ty:.4f} {tz:.4f}"
            objs.append(f'    <object id="{obj_id}" p:UUID="{uuid4()}" type="model">\n      <components>\n'
                        + f'        <component p:path="{part}" objectid="{mesh_id}" p:UUID="{uuid4()}"'
                        + f' transform="{IDENTITY} 0 0 0"/>\n      </components>\n    </object>\n')
            builds.append(f'    <item objectid="{obj_id}" p:UUID="{uuid4()}" transform="{tf}" printable="1"/>\n')
            name = ob["name"].replace('"', "'")
            settings.append(f'  <object id="{obj_id}">\n    <metadata key="name" value="{name}"/>\n'
                            + f'    <metadata key="extruder" value="{ob["extruder"]}"/>\n'
                            + f'    <part id="{mesh_id}" subtype="normal_part">\n'
                            + f'      <metadata key="name" value="{name}"/>\n'
                            + f'      <metadata key="extruder" value="{ob["extruder"]}"/>\n'
                            + '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                            + "    </part>\n  </object>\n")
            instances.append(f'    <model_instance>\n      <metadata key="object_id" value="{obj_id}"/>\n'
                             + '      <metadata key="instance_id" value="0"/>\n'
                             + f'      <metadata key="identify_id" value="{k + 1}"/>\n    </model_instance>\n')
            assemble.append(f'    <assemble_item object_id="{obj_id}" instance_id="0" transform="{tf}" offset="0 0 0"/>\n')

        zf.writestr("[Content_Types].xml",
                    XML + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                    + '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                    + '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
                    + "</Types>\n")
        zf.writestr("_rels/.rels", rel_open
                    + f'  <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="{REL_TYPE}"/>\n</Relationships>\n')
        zf.writestr("3D/_rels/3dmodel.model.rels", rel_open + "".join(rels) + "</Relationships>\n")
        zf.writestr("3D/3dmodel.model", _model_head() + '  <metadata name="Title">aprilcylinder</metadata>\n'
                    + "  <resources>\n" + "".join(objs) + f'  </resources>\n  <build p:UUID="{uuid4()}">\n'
                    + "".join(builds) + "  </build>\n</model>\n")
        zf.writestr("Metadata/model_settings.config", XML + "<config>\n" + "".join(settings)
                    + '  <plate>\n    <metadata key="plater_id" value="1"/>\n    <metadata key="plater_name" value=""/>\n'
                    + '    <metadata key="locked" value="false"/>\n' + "".join(instances) + "  </plate>\n"
                    + "  <assemble>\n" + "".join(assemble) + "  </assemble>\n</config>\n")
        zf.writestr("Metadata/project_settings.config", json.dumps({
            "version": "02.00.00.00", "printer_technology": "FFF",
            "filament_colour": ["#000000", "#FFFFFF"], "filament_type": ["PLA", "PLA"],
            "nozzle_diameter": ["0.4"]}, indent=4) + "\n")


def write_3mf(path, mesh: Mesh, name: str, body: str = "black", position=(128.0, 128.0)) -> None:
    """Single-object 2-colour 3MF. body = "black" (filament 1) or "white" (filament 2)."""
    body_filament = 2 if body == "white" else 1
    paint_code = PAINT_FILAMENT[1 if body == "white" else 2]
    lo, _ = mesh.bounds()
    write_3mf_objects(path, [{"name": name, "mesh_xml": mesh_xml(mesh, paint_code), "extruder": body_filament,
                              "position": (position[0], position[1], -float(lo[2]))}])


def write_stl(path, mesh: Mesh) -> None:
    V = np.asarray(mesh.verts)
    with open(path, "wb") as f:
        f.write(b"aprilcylinder".ljust(80, b" "))
        f.write(struct.pack("<I", len(mesh.tris)))
        for a, b, c, _ in mesh.tris:
            n = np.cross(V[b] - V[a], V[c] - V[a])
            n = n / (np.linalg.norm(n) + 1e-12)
            f.write(struct.pack("<12fH", *n, *V[a], *V[b], *V[c], 0))


def load_stl(path) -> np.ndarray:
    """Binary STL -> (n, 3, 3) float array of triangle vertices."""
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + 50 * n != len(data):
        raise ValueError(f"{path}: only binary STL is supported (export the STL as binary)")
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n, offset=84)
    return rec["v"].reshape(-1, 3, 3).astype(float)


def reorient_up(tris: np.ndarray, up_axis: str) -> np.ndarray:
    """Rotate so that the given STL axis becomes +Z (right-handed rotation, no mirroring)."""
    if up_axis == "y":   # rotate +90 deg about X: (x, y, z) -> (x, -z, y)
        return np.stack([tris[..., 0], -tris[..., 2], tris[..., 1]], axis=-1)
    if up_axis == "x":   # rotate -90 deg about Y: (x, y, z) -> (-z, y, x)
        return np.stack([-tris[..., 2], tris[..., 1], tris[..., 0]], axis=-1)
    return tris


# ---------------------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------------------
def _rot(ax: int, deg: float) -> np.ndarray:
    v = np.zeros(3)
    v[ax] = math.radians(deg)
    return cv2.Rodrigues(v)[0]


def render_preview(path, mesh: Mesh, config: dict, lines: list[str], body: str = "black", tile: int = 640) -> None:
    """3-view preview (two sides and from below) with marker IDs, axes and a text footer."""
    V0 = np.asarray(mesh.verts, float)
    lo, hi = V0.min(0), V0.max(0)
    center = (lo + hi) / 2.0
    V = V0 - center
    ext = hi - lo
    size = float(np.linalg.norm([max(ext[0], ext[1]), ext[2]]))
    f = tile * 2.2
    K = np.array([[f, 0, tile / 2], [0, f, tile / 2], [0, 0, 1.0]])
    tv = np.array([0.0, 0.0, size * 3.0])
    views = [("side", _rot(0, 112) @ _rot(2, 25)), ("other side", _rot(0, 112) @ _rot(2, 205)),
             ("from below", _rot(0, 35) @ _rot(2, 25))]
    light = np.array([0.3, -0.5, -0.8])
    light /= np.linalg.norm(light)
    painted_is_white = body != "white"
    tiles = []
    for label, R in views:
        img = np.full((tile, tile, 3), 240, np.uint8)
        rvec = cv2.Rodrigues(R)[0]
        pv = cv2.projectPoints(V, rvec, tv, K, None)[0].reshape(-1, 2)
        Vc = (R @ V.T).T + tv
        for a, b, c, painted in sorted(mesh.tris, key=lambda q: -(Vc[q[0]][2] + Vc[q[1]][2] + Vc[q[2]][2])):
            nrm = np.cross(Vc[b] - Vc[a], Vc[c] - Vc[a])
            if nrm @ Vc[a] >= 0:
                continue
            shade = 0.78 + 0.22 * max(0.0, float(nrm @ light) / (np.linalg.norm(nrm) + 1e-12))
            white = painted == painted_is_white
            g = int(250 * shade) if white else int(35 * shade)
            cv2.fillConvexPoly(img, np.round(pv[[a, b, c]] * 16).astype(np.int32), (g, g, g), cv2.LINE_AA, 4)
        L = size * 0.5
        axes = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], float) - center
        ap = cv2.projectPoints(axes, rvec, tv, K, None)[0].reshape(-1, 2)
        for i, (name, col) in enumerate((("X", (0, 0, 230)), ("Y", (0, 170, 0)), ("Z", (230, 0, 0))), start=1):
            p0, p1 = tuple(int(v) for v in ap[0]), tuple(int(v) for v in ap[i])
            cv2.line(img, p0, p1, col, 3, cv2.LINE_AA)
            cv2.putText(img, name, (p1[0] + 6, p1[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        for mk in config["markers"]:
            c3 = np.asarray(mk["corners_mm"], float) - center
            if (R @ np.asarray(mk["normal"], float)) @ (R @ c3.mean(0) + tv) >= 0:
                continue
            p = cv2.projectPoints(c3.mean(0).reshape(1, 3), rvec, tv, K, None)[0].ravel()
            text = str(mk["id"])
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            x, y = int(p[0] - tw / 2), int(p[1] + th / 2)
            cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + 4), (255, 255, 255), -1)
            cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + 4), (0, 120, 255), 1)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 90, 200), 1, cv2.LINE_AA)
        cv2.putText(img, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)
        tiles.append(img)
    footer = np.full((24 + 34 * len(lines), tile * len(tiles), 3), 240, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(footer, line, (16, 30 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 60, 60), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", np.vstack([np.hstack(tiles), footer]))
    Path(path).write_bytes(buf.tobytes())   # works with non-ASCII paths on Windows
