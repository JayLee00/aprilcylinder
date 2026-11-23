"""Track a cylinder target with an Intel RealSense colour stream.

    pip install pyrealsense2
    python examples/track_realsense.py models/band8_d50_h120

Prints the angle about the axis and the axis base point (camera frame, mm). Press z to make the
current orientation 0 deg, q to quit.
"""
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

from aprilcylinder import CylinderTracker

model = sys.argv[1]
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipe.start(cfg)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
tracker = CylinderTracker(model, {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                                  "dist_coeffs": list(intr.coeffs)}, fast=True)
try:
    while True:
        frame = np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data())
        st = tracker.process(frame)
        if st.success:
            x, y, z = st.axis_origin_mm
            print(f"angle {st.angle_deg:7.2f} deg  unwrapped {st.angle_unwrapped_deg:9.2f}  "
                  f"tilt {st.tilt_deg:5.2f}  base ({x:6.1f}, {y:6.1f}, {z:6.1f}) mm  tags {st.n_tags}")
        cv2.imshow("aprilcylinder", st.raw.get("debug_viz", frame) if st.raw else frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("z") and st.success:
            tracker.set_reference(state=st)
finally:
    pipe.stop()
