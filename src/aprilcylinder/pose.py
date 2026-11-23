"""Rotation angle about the cylinder axis and axis position, from aprilcube poses.

Every marker has its own ID and its corners are fixed in the target frame, so a single image gives
the full 6-DoF pose of the target. The angle about the axis is therefore absolute: it does not
depend on earlier frames, and it is recovered correctly after occlusions.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AXIS = (0.0, 0.0, 1.0)


def _quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), numerically safe for any angle."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(1.0 + t)
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k]))
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def twist_angle(R_ref, R, axis=AXIS) -> float:
    """Signed angle (rad, in (-pi, pi]) that R is turned from R_ref about `axis` (in the target frame).

    Swing-twist decomposition: only the rotation about the axis is returned, so a tilt of the axis
    does not leak into the angle.
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    q = _quat(np.asarray(R_ref, float).T @ np.asarray(R, float))
    ang = 2.0 * math.atan2(float(q[1:] @ a), float(q[0]))
    return (ang + math.pi) % (2.0 * math.pi) - math.pi


def swing_angle(R_ref, R, axis=AXIS) -> float:
    """Angle (rad) between the axis direction of R_ref and that of R."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c = float(np.clip((np.asarray(R_ref) @ a) @ (np.asarray(R) @ a), -1.0, 1.0))
    return math.acos(c)


@dataclass
class CylinderState:
    success: bool
    angle_deg: float | None = None              # [0, 360) about the axis, relative to the reference
    angle_unwrapped_deg: float | None = None    # keeps counting past 360 (needs < 180 deg between frames)
    axis_origin_mm: np.ndarray | None = None    # axis point at the bottom face, in the output frame
    axis_dir: np.ndarray | None = None          # unit axis direction (+Z of the target) in the output frame
    tilt_deg: float | None = None               # angle between the current axis and the reference axis
    T: np.ndarray | None = None                 # 4x4 target pose in the output frame (translation in mm)
    n_tags: int = 0
    tag_ids: list = field(default_factory=list)
    reproj_error: float | None = None
    raw: dict | None = None                     # aprilcube result

    def point_on_axis(self, z_mm: float) -> np.ndarray | None:
        """Point on the axis at height z_mm above the bottom face, in the output frame."""
        if not self.success:
            return None
        return self.axis_origin_mm + self.axis_dir * z_mm


class CylinderTracker:
    """Track the angle about the axis and the axis position of an aprilcylinder target.

    Args:
        model: model folder (with config.json) or the config.json path.
        intrinsics: camera intrinsics, as accepted by ``aprilcube.detector``
            (dict with fx, fy, cx, cy [, dist_coeffs], a 3x3 matrix, or a calibration JSON path).
        extrinsic: optional 4x4 T_world_cam with translation in **mm**. If given, all outputs are in
            the world frame; otherwise they are in the camera frame.
        reference: what 0 deg means.
            "first"  - the first successful detection (default),
            "frame"  - the output frame itself; use this when the world Z axis is the cylinder axis,
                       then the angle is measured from the world X axis,
            3x3 array - an explicit reference orientation of the target in the output frame.
        axis: the cylinder axis in the target frame (default +Z, which is what this package generates).
        fast, enable_filter, dist_coeffs: passed to ``aprilcube.detector``.
    """

    def __init__(self, model, intrinsics, *, extrinsic=None, reference="first", axis=AXIS,
                 fast: bool = False, enable_filter: bool = True, dist_coeffs=None):
        import aprilcube

        path = Path(model)
        if path.is_dir():
            path = path / "config.json"
        self.config = json.loads(path.read_text(encoding="utf-8"))
        self.detector = aprilcube.detector(str(path), intrinsics, fast=fast, enable_filter=enable_filter,
                                           dist_coeffs=dist_coeffs)
        self.extrinsic = None if extrinsic is None else np.asarray(extrinsic, float)
        self.axis = np.asarray(axis, float) / np.linalg.norm(axis)
        self._ref_mode = reference if isinstance(reference, str) else "matrix"
        self.R_ref = None if isinstance(reference, str) else np.asarray(reference, float)
        if self._ref_mode == "frame":
            self.R_ref = np.eye(3)
        if self._ref_mode not in ("first", "frame", "matrix"):
            raise ValueError("reference must be 'first', 'frame' or a 3x3 rotation matrix")
        self._last = None
        self._total = 0.0

    # -- reference ----------------------------------------------------------------------
    def set_reference(self, R=None, state: CylinderState | None = None) -> None:
        """Make R (or the pose in `state`, or the last good pose) the 0 deg orientation."""
        if R is None:
            src = state if state is not None else self._last_state
            if src is None or not src.success:
                raise RuntimeError("no successful detection to use as reference")
            R = src.T[:3, :3]
        self.R_ref = np.asarray(R, float).copy()
        self._last, self._total = None, 0.0

    def reset_unwrap(self) -> None:
        self._last, self._total = None, 0.0

    _last_state: CylinderState | None = None

    # -- per frame ------------------------------------------------------------------------
    def from_result(self, result: dict) -> CylinderState:
        """Convert an aprilcube ``process_frame`` result into a CylinderState."""
        if not result.get("success"):
            return CylinderState(False, n_tags=int(result.get("n_tags", 0) or 0), raw=result)
        T = np.asarray(result["T"], float).copy()          # camera frame, mm
        if self.extrinsic is not None:
            T = self.extrinsic @ T
        R = T[:3, :3]
        if self.R_ref is None:                              # reference = "first"
            self.R_ref = R.copy()
        ang = twist_angle(self.R_ref, R, self.axis)
        if self._last is None:
            self._total = ang
        else:
            d = (ang - self._last + math.pi) % (2.0 * math.pi) - math.pi
            self._total += d
        self._last = ang
        state = CylinderState(
            success=True,
            angle_deg=math.degrees(ang) % 360.0,
            angle_unwrapped_deg=math.degrees(self._total),
            axis_origin_mm=T[:3, 3].copy(),
            axis_dir=R @ self.axis,
            tilt_deg=math.degrees(swing_angle(self.R_ref, R, self.axis)),
            T=T,
            n_tags=int(result.get("n_tags", 0) or 0),
            tag_ids=[int(i) for i in result.get("tag_ids", [])],
            reproj_error=float(result.get("reproj_error", float("nan"))),
            raw=result,
        )
        self._last_state = state
        return state

    def process(self, frame, timestamp: float | None = None) -> CylinderState:
        """Detect the target in a BGR (or grayscale) frame."""
        return self.from_result(self.detector.process_frame(frame, timestamp=timestamp))
