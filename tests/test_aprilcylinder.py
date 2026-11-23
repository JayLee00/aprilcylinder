import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from aprilcylinder.cylinder import Section, TargetSpec, analytic_volume, build, generate, load_spec, simple_spec
from aprilcylinder.merge import merge
from aprilcylinder.mesh import Mesh, write_stl
from aprilcylinder.paint_stl import paint_stl
from aprilcylinder.pose import CylinderTracker, twist_angle
from aprilcylinder.synth import load_painted_mesh, render, rot, selftest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("kw", [
    dict(diameter=50, height=120, sides=8, face=22, bands="bottom"),
    dict(diameter=40, height=100, sides=6, bands="both", bottom_marker=True, body="black"),
    dict(diameter=30, height=80, sides=8, face=20, band_height=40, bands="top"),
    dict(diameter=20, height=50, sides=5, bands="none", bottom_marker=True),
])
def test_mesh_closed_and_volume(kw):
    geo = {k: kw.pop(k) for k in list(kw) if k in ("diameter", "height", "sides", "face", "bands", "band_height")}
    spec = simple_spec(**geo, **kw)
    mesh, cfg, _ = build(spec)
    assert mesh.is_closed()
    assert mesh.volume() == pytest.approx(analytic_volume(spec), rel=1e-6)
    assert len(cfg["markers"]) == len(set(cfg["tag_ids"]))


def test_marker_corners_on_faces():
    spec = simple_spec(50, 120, sides=8, face=22)
    _, cfg, _ = build(spec)
    for mk in cfg["markers"]:
        c = np.asarray(mk["corners_mm"])
        n = np.asarray(mk["normal"])
        # planar, square, and the corner order TL, TR, BR, BL makes (TR-TL) x (BL-TL) point inward
        assert np.ptp(c @ n) < 1e-4
        sides = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
        assert max(sides) - min(sides) < 1e-4 and sides[0] == pytest.approx(mk["tag_size_mm"], abs=1e-4)
        assert np.cross(c[1] - c[0], c[3] - c[0]) @ n < 0
        if mk["face"].startswith("s0"):
            assert c[:, 2].min() >= -1e-9 and c[:, 2].max() <= 22 + 1e-9
            assert c @ n == pytest.approx(np.full(4, 22 / (2 * math.tan(math.pi / 8))), abs=1e-4)


def test_overlapping_sections_rejected():
    spec = TargetSpec(sections=[Section("polygon", 20, sides=8, apothem=25.5), Section("cylinder", 50, radius=26)])
    with pytest.raises(ValueError):
        build(spec)


def test_examples_build():
    for f in EXAMPLES.glob("*.yaml"):
        spec, _ = load_spec(f)
        mesh, cfg, _ = build(spec)
        assert mesh.is_closed(), f.name
        assert mesh.volume() == pytest.approx(analytic_volume(spec), rel=1e-6)


def test_3mf_paint_and_body(tmp_path):
    for body, code, extruder in (("white", "4", "2"), ("black", "8", "1")):
        spec = simple_spec(40, 80, sides=8, body=body, name="t")
        generate(spec, tmp_path / body, preview=False)
        with zipfile.ZipFile(tmp_path / body / "t.3mf") as zf:
            xml = zf.read("3D/Objects/object_1.model").decode()
            settings = zf.read("Metadata/model_settings.config").decode()
        assert f'paint_color="{code}"' in xml
        assert f'key="extruder" value="{extruder}"' in settings


def test_twist_angle():
    R0 = rot(0, 30) @ rot(1, -20)
    for deg in (-170, -45, 0, 10, 90, 179):
        assert math.degrees(twist_angle(R0, R0 @ rot(2, deg))) == pytest.approx(deg, abs=1e-6)
    # a tilt of the axis does not leak into the twist
    assert math.degrees(twist_angle(R0, R0 @ rot(0, 15))) == pytest.approx(0.0, abs=1e-6)


def test_paint_stl_keeps_shape(tmp_path):
    # an octagonal prism whose side faces are 20 mm squares (two triangles each)
    spec = TargetSpec(sections=[Section("polygon", 20.0, sides=8, apothem=20 / (2 * math.tan(math.pi / 8)), markers=False)],
                      top_marker=False, bottom_marker=False)
    mesh, _, _ = build(spec, allow_no_markers=True)
    stl = tmp_path / "prism.stl"
    write_stl(stl, mesh)
    cfg = paint_stl(stl, tmp_path / "out", first_id=10, margin_mm=1.0, preview=False)
    assert len(cfg["markers"]) == 8
    assert cfg["markers"][0]["tag_size_mm"] == pytest.approx(18.0)
    V, T = load_painted_mesh(tmp_path / "out")
    m = Mesh()
    for a, b, c, white in T:
        m.tri(V[a], V[b], V[c], np.cross(V[b] - V[a], V[c] - V[a]))
    assert m.is_closed()
    assert m.volume() == pytest.approx(mesh.volume(), rel=1e-6)


def test_merge(tmp_path):
    for i, body in enumerate(("white", "black")):
        generate(simple_spec(30, 60, sides=6, body=body, first_id=20 * i, name=f"m{i}"), tmp_path / f"m{i}", preview=False)
    stl = tmp_path / "plain.stl"
    mesh, _, _ = build(TargetSpec(sections=[Section("cylinder", 10, radius=10)], top_marker=False), allow_no_markers=True)
    write_stl(stl, mesh)
    items = merge([tmp_path / "m0", tmp_path / "m1", stl], tmp_path / "all.3mf", stl_filament=2)
    assert [it["extruder"] for it in items] == [2, 1, 2]
    with zipfile.ZipFile(tmp_path / "all.3mf") as zf:
        assert len([n for n in zf.namelist() if n.startswith("3D/Objects/")]) == 3


def test_absolute_angle_single_images(tmp_path):
    """Each image is processed by a fresh tracker: the angle must come from that image alone."""
    generate(simple_spec(50, 120, sides=8, face=22, first_id=100, name="c"), tmp_path / "c", preview=False)
    res = selftest(tmp_path / "c", distance_mm=400, n=12, views=("side", "oblique", "top"))
    for view, r in res.items():
        assert r["detected"] >= 11, (view, r)
        assert r["angle_err_mean_deg"] < 1.5, (view, r)
        assert r["angle_err_max_deg"] < 4.0, (view, r)


def test_axis_position(tmp_path):
    generate(simple_spec(50, 120, sides=8, face=22, name="c"), tmp_path / "c", preview=False)
    V, T = load_painted_mesh(tmp_path / "c")
    K = np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1.0]])
    R = rot(0, 125) @ rot(2, 37)
    t = np.array([20.0, -10.0, 420.0])
    st = CylinderTracker(tmp_path / "c", K, reference=R, enable_filter=False).process(render(V, T, K, R, t))
    assert st.success
    assert st.angle_deg == pytest.approx(0.0, abs=1.5) or st.angle_deg == pytest.approx(360.0, abs=1.5)
    assert np.linalg.norm(st.axis_origin_mm - t) < 5.0
    assert math.degrees(math.acos(float(st.axis_dir @ (R @ [0, 0, 1])))) < 1.5
    top = st.point_on_axis(120.0)
    assert np.linalg.norm(top - (t + R @ [0, 0, 120.0])) < 6.0


def test_config_is_readable_by_aprilcube(tmp_path):
    import aprilcube
    generate(simple_spec(40, 90, sides=6, name="c"), tmp_path / "c", preview=False)
    det = aprilcube.detector(str(tmp_path / "c"), {"fx": 900, "fy": 900, "cx": 640, "cy": 360})
    cfg = json.loads((tmp_path / "c" / "config.json").read_text())
    assert sorted(det.valid_ids) == sorted(cfg["tag_ids"])
