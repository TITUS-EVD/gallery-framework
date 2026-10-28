"""
3D points (tracks, vertices, spacepoints, showers) projected into TPC views,
in the same coordinates as wire images and hits (titus.display.layout).

A point in a TPC lands in each plane of that TPC: at its continuous wire
coordinate, and at the tick charge from its x arrives (Plane.x_to_ticks).
Points of a path are split into segments per TPC, so a track crossing the
cathode is drawn in both drift volumes' blocks.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Segment:
    plane_index: int    # TITUS plane index of the block drawn in
    x: np.ndarray       # view coordinates
    y: np.ndarray


def ticks_per_ns(geom):
    return 1. / (geom.detector.timing.tpc_tick_period * 1000.)


def project_points(layout, xyz, tick_shift=0.):
    """
    View coordinates of 3D points, per block of the layout: {plane index:
    (point indices, x, y)} for the blocks that have points. tick_shift
    (ticks) is added to the drift time, e.g. a T0 that was removed from the
    points' x.
    """
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    detector = layout.geom.detector
    tpcs = detector.tpc_of(xyz)
    result = {}
    for block in layout.blocks:
        plane = detector.planes[block.plane_index]
        index = np.flatnonzero((tpcs[:, 0] == plane.id.cryostat) & (tpcs[:, 1] == plane.id.tpc))
        if len(index) == 0:
            continue
        points = xyz[index]
        x = layout.wire_to_view(block.plane_index, plane.wire_coordinate(points))
        y = layout.tick_to_view(block.plane_index, plane.x_to_ticks(points[:, 0]) + tick_shift)
        result[block.plane_index] = (index, x, y)
    return result


def project_path(layout, xyz, tick_shift=0.) -> List[Segment]:
    """
    A path (e.g. a track trajectory) as segments of consecutive points in
    the same TPC
    """
    segments = []
    for plane_index, (index, x, y) in project_points(layout, xyz, tick_shift).items():
        breaks = np.flatnonzero(np.diff(index) != 1) + 1
        for run in np.split(np.arange(len(index)), breaks):
            if len(run) > 1:
                segments.append(Segment(plane_index, x[run], y[run]))
    return segments
