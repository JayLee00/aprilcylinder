"""Command line interface: aprilcyl <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .mesh import DICTS


def _cmd_generate(a):
    from .cylinder import generate, load_spec
    spec, out = load_spec(a.spec)
    out = a.output or out or Path(a.spec).stem
    if a.name:
        spec.name = a.name
    elif spec.name == "aprilcylinder":
        spec.name = Path(out).name
    generate(spec, out, preview=not a.no_preview)


def _cmd_cylinder(a):
    from .cylinder import generate, simple_spec
    spec = simple_spec(a.diameter, a.height, sides=a.sides, face=a.face, band_height=a.band_height,
                       bands=a.bands, dictionary=a.dict, first_id=a.first_id, margin_mm=a.margin, body=a.body,
                       segments=a.segments, phase_deg=a.phase, top_marker=not a.no_top_marker,
                       bottom_marker=a.bottom_marker, top_square_mm=a.top_square, name=Path(a.output).name)
    generate(spec, a.output, preview=not a.no_preview)


def _cmd_paint(a):
    from .paint_stl import paint_stl
    paint_stl(a.stl, a.output, up_axis=a.up_axis, dictionary=a.dict, first_id=a.first_id, margin_mm=a.margin,
              body=a.body, min_size=a.min_size, preview=not a.no_preview)


def _cmd_merge(a):
    from .merge import merge
    w, d = (float(x) for x in a.plate.lower().split("x"))
    merge(a.inputs, a.output, plate=(w, d), gap=a.gap, stl_filament=a.stl_filament, stl_up_axis=a.stl_up_axis)


def _cmd_selftest(a):
    from .synth import selftest
    for dist in a.distance:
        res = selftest(a.model, distance_mm=dist, n=a.frames, fx=a.fx)
        for view, r in res.items():
            print(f"{dist:6.0f} mm  {view:8s} detected {r['detected']:3d}/{r['frames']}  "
                  f"tags/frame {r['tags_per_frame']:.1f}  angle error mean {r['angle_err_mean_deg']:.2f} deg, "
                  f"max {r['angle_err_max_deg']:.2f} deg")


def _cmd_view(a):
    from .viewer import demo_fixture, fixture_from_stl, scene_payload, write_viewer
    fixture, name = None, None
    if a.fixture == "demo":
        fixture, seat = demo_fixture()
        name = "demo base"
        if a.seat is None:
            a.seat = seat
    elif a.fixture:
        fixture = fixture_from_stl(a.fixture, a.fixture_up_axis, a.fixture_axis)
        name = Path(a.fixture).stem
    scenes = [scene_payload(m, fixture=fixture, seat_z=a.seat or 0.0, fixture_name=name) for m in a.models]
    out = write_viewer(scenes, a.output, title=a.title)
    print(f"wrote {out} (open it in a browser; needs internet for three.js)")


def _cmd_obj(a):
    from .viewer import export_obj
    out = export_obj(a.model, a.output)
    print(f"wrote {out} and {out.with_suffix('.mtl')}")


def _cmd_track(a):
    import cv2
    import numpy as np
    from .pose import CylinderTracker

    cap = cv2.VideoCapture(a.camera)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"cannot read camera {a.camera}")
    if a.intrinsics:
        intr = a.intrinsics
    else:
        h, w = frame.shape[:2]
        f = a.fx or 0.9 * w
        intr = {"fx": f, "fy": f, "cx": w / 2, "cy": h / 2}
        print(f"no --intrinsics: assuming fx = fy = {f:.0f} px (angles are fine, distances are approximate)")
    trk = CylinderTracker(a.model, intr, fast=True)
    print("keys: z = set the current pose as 0 deg, q = quit")
    while ok:
        st = trk.process(frame)
        txt = "no detection"
        if st.success:
            p = st.axis_origin_mm
            txt = (f"angle {st.angle_deg:7.2f} deg  turns {st.angle_unwrapped_deg:9.2f} deg  tilt {st.tilt_deg:5.1f} deg  "
                   f"axis base ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm  tags {st.n_tags}")
            viz = st.raw.get("debug_viz") if st.raw else None
            if viz is not None:
                frame = viz
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("aprilcylinder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("z") and st.success:
            trk.set_reference(state=st)
        ok, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def main(argv=None):
    p = argparse.ArgumentParser(prog="aprilcyl", description="Cylindrical ArUco fiducial targets built on aprilcube")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, body_default="white"):
        sp.add_argument("--dict", default="4x4_1000", choices=sorted(DICTS))
        sp.add_argument("--first-id", type=int, default=0, help="first marker ID (keep targets in one scene distinct)")
        sp.add_argument("--margin", type=float, default=1.0, help="white margin around each marker, mm (default 1)")
        sp.add_argument("--body", choices=["white", "black"], default=body_default, help="body colour")
        sp.add_argument("--no-preview", action="store_true", help="skip thumbnail.png")

    g = sub.add_parser("generate", help="generate a target from a YAML spec")
    g.add_argument("spec")
    g.add_argument("-o", "--output")
    g.add_argument("--name")
    g.add_argument("--no-preview", action="store_true")
    g.set_defaults(func=_cmd_generate)

    c = sub.add_parser("cylinder", help="cylinder with a polygonal marker band (quick form)")
    c.add_argument("--diameter", type=float, required=True, help="cylinder diameter, mm")
    c.add_argument("--height", type=float, required=True, help="total height including bands, mm")
    c.add_argument("--sides", type=int, default=8, help="band faces (default 8)")
    c.add_argument("--face", type=float, help="band face width, mm (default: band clears the cylinder by 1.5 mm)")
    c.add_argument("--band-height", type=float, help="band height, mm (default: = face width)")
    c.add_argument("--bands", choices=["bottom", "top", "both", "none"], default="bottom")
    c.add_argument("--no-top-marker", action="store_true")
    c.add_argument("--bottom-marker", action="store_true")
    c.add_argument("--top-square", type=float, help="top marker square incl. margin, mm (default: largest that fits)")
    c.add_argument("--phase", type=float, default=0.0, help="rotate the band so face 0 points at this angle, deg")
    c.add_argument("--segments", type=int, default=128, help="facets of the round section")
    c.add_argument("-o", "--output", required=True)
    common(c)
    c.set_defaults(func=_cmd_cylinder)

    s = sub.add_parser("paint-stl", help="put markers on the flat square faces of an STL part")
    s.add_argument("stl")
    s.add_argument("--up-axis", choices=["x", "y", "z"], default="z", help="STL axis that is the part axis")
    s.add_argument("--min-size", type=float, default=15.0, help="ignore square faces smaller than this, mm")
    s.add_argument("-o", "--output", required=True)
    common(s)
    s.set_defaults(func=_cmd_paint)

    m = sub.add_parser("merge", help="combine targets and STL parts into one 3MF")
    m.add_argument("inputs", nargs="+", help="model folders, .3mf files made by this tool, or .stl files")
    m.add_argument("-o", "--output", required=True)
    m.add_argument("--plate", default="256x256", help="plate size WxD in mm for the initial layout")
    m.add_argument("--gap", type=float, default=10.0)
    m.add_argument("--stl-filament", type=int, choices=[1, 2], default=2, help="colour of plain STL parts (1 black, 2 white)")
    m.add_argument("--stl-up-axis", choices=["x", "y", "z"], default="z")
    m.set_defaults(func=_cmd_merge)

    t = sub.add_parser("selftest", help="render the model synthetically and check detection and angle error")
    t.add_argument("model")
    t.add_argument("--distance", type=float, nargs="+", default=[400.0, 700.0], help="camera distances, mm")
    t.add_argument("--frames", type=int, default=36)
    t.add_argument("--fx", type=float, default=900.0, help="focal length in px of the virtual 1280x720 camera")
    t.set_defaults(func=_cmd_selftest)

    v = sub.add_parser("view", help="interactive 3D viewer (HTML) with the target turning about Z")
    v.add_argument("models", nargs="+", help="model folders")
    v.add_argument("-o", "--output", default="viewer.html")
    v.add_argument("--title", default="aprilcylinder viewer")
    v.add_argument("--fixture", help="fixed part to show under the target: an STL path, or 'demo'")
    v.add_argument("--fixture-up-axis", choices=["x", "y", "z"], default="z")
    v.add_argument("--fixture-axis", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"),
                   help="where the rotation axis is in the fixture (after --fixture-up-axis), mm")
    v.add_argument("--seat", type=float, help="height of the target bottom above the fixture's z = 0, mm")
    v.set_defaults(func=_cmd_view)

    o = sub.add_parser("export-obj", help="export a target as OBJ + MTL (black / white materials) for CAD tools")
    o.add_argument("model")
    o.add_argument("-o", "--output", required=True)
    o.set_defaults(func=_cmd_obj)

    k = sub.add_parser("track", help="live angle / axis tracking from a webcam")
    k.add_argument("model")
    k.add_argument("--camera", type=int, default=0)
    k.add_argument("--intrinsics", help="calibration JSON with camera_matrix [and dist_coeffs]")
    k.add_argument("--fx", type=float, help="focal length in px if no calibration file")
    k.set_defaults(func=_cmd_track)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
