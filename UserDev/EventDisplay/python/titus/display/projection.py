"""
3D points (tracks, vertices, spacepoints, showers) projected into TPC views,
in the same coordinates as wire images and hits (titus.display.layout).

A point lands in each plane of its TPC: at its continuous wire coordinate,
and at the tick charge from its x arrives (Plane.x_to_ticks). Points of a
path are split into segments per TPC, so a track crossing the cathode is
drawn in both drift volumes' blocks.

Which TPC a point belongs to is not always where it is: reconstruction
places charge assuming t0 = 0, so a cosmic that arrived before or after the
trigger appears behind the anode or past the cathode, in the other TPC's
volume. The TPC is best taken from the point's hits (tpcs_from_hits);
without them, readout_tpcs picks the TPC whose readout window contains the
point. Pandora also moves T0-tagged objects to where their charge was, off
their hits by the T0.

Objects can be drawn as stored (position_tpcs, no shift), or corrected to
lie on their hits using only what the reconstruction stored (its T0s and
track-hit associations; see track_correction). Corrections are reported per
object, so a display can show what was changed: a wrong T0 or a bad fit
still draws off its hits.
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


def readout_tpcs(detector, xyz, n_ticks, tick_shift=0.):
    """
    (cryostat, tpc) per point: the TPC containing it, else the TPC whose
    readout window (ticks 0 to n_ticks) and y-z extent contain it, e.g. an
    early cosmic placed behind the anode. Points past the cathode of an
    untagged late cosmic can't be told apart from points in the other TPC
    without their hits. (-1, -1) if none.
    """
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    tick_shift = np.broadcast_to(np.asarray(tick_shift, dtype=float), (len(xyz),))
    tpcs = detector.tpc_of(xyz)
    for i in np.flatnonzero(tpcs[:, 0] < 0):
        for tpc in detector.tpcs:
            lo, hi = tpc.bounds.min, tpc.bounds.max
            if not np.all((xyz[i, 1:] >= lo[1:]) & (xyz[i, 1:] <= hi[1:])):
                continue
            tick = tpc.planes[-1].x_to_ticks(xyz[i, 0]) + tick_shift[i]
            if 0 <= tick < n_ticks:
                tpcs[i] = tpc.id
                break
    return tpcs


def position_tpcs(detector, xyz):
    """
    (cryostat, tpc) per point by position only: the TPC containing it, else
    the nearest one. Used to draw objects as stored, without corrections.
    """
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    tpcs = detector.tpc_of(xyz)
    outside = np.flatnonzero(tpcs[:, 0] < 0)
    if len(outside):
        boxes = detector.tpcs
        distance = np.stack([np.linalg.norm(np.maximum(0, np.maximum(t.bounds.min - xyz[outside],
                                                                     xyz[outside] - t.bounds.max)), axis=1)
                             for t in boxes], axis=1)
        tpcs[outside] = [boxes[i].id for i in distance.argmin(axis=1)]
    return tpcs


def tpcs_from_hits(n_points, point_index, hit_rows, hits):
    """
    (cryostat, tpc) per trajectory point from the hits associated with it
    (point_index: TrackHitMeta index per associated hit). Points without a
    hit take the TPC of the nearest point with one; all (-1, -1) if the
    track has no usable hits.
    """
    tpcs = np.full((n_points, 2), -1, dtype=int)
    point_index = np.asarray(point_index)
    ok = (point_index >= 0) & (point_index < n_points)
    if not ok.any():
        return tpcs
    with_hit = point_index[ok]
    tpcs[with_hit, 0] = hits.cryostat[np.asarray(hit_rows)[ok]]
    tpcs[with_hit, 1] = hits.tpc[np.asarray(hit_rows)[ok]]
    known = np.unique(with_hit)
    nearest = known[np.abs(np.arange(n_points)[:, None] - known[None, :]).argmin(axis=1)]
    return tpcs[nearest]


def project_points(layout, xyz, tick_shift=0., tpcs=None):
    """
    View coordinates of 3D points, per block of the layout: {plane index:
    (point indices, x, y)} for the blocks that have points. tick_shift
    (ticks, one or per point) is added to the drift time, e.g. a T0 that was
    removed from the points' x. tpcs: (cryostat, tpc) per point; by default
    readout_tpcs.
    """
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    tick_shift = np.broadcast_to(np.asarray(tick_shift, dtype=float), (len(xyz),))
    detector = layout.geom.detector
    if tpcs is None:
        tpcs = readout_tpcs(detector, xyz, layout.n_ticks, tick_shift)
    tpcs = np.asarray(tpcs).reshape(-1, 2)
    result = {}
    for block in layout.blocks:
        plane = detector.planes[block.plane_index]
        index = np.flatnonzero((tpcs[:, 0] == plane.id.cryostat) & (tpcs[:, 1] == plane.id.tpc))
        if len(index) == 0:
            continue
        points = xyz[index]
        x = layout.wire_to_view(block.plane_index, plane.wire_coordinate(points))
        y = layout.tick_to_view(block.plane_index, plane.x_to_ticks(points[:, 0]) + tick_shift[index])
        result[block.plane_index] = (index, x, y)
    return result


def project_path(layout, xyz, tick_shift=0., tpcs=None) -> List[Segment]:
    """
    A path (e.g. a track trajectory) as segments of consecutive points in
    the same TPC. tpcs: see project_points.
    """
    segments = []
    for plane_index, (index, x, y) in project_points(layout, xyz, tick_shift, tpcs).items():
        breaks = np.flatnonzero(np.diff(index) != 1) + 1
        for run in np.split(np.arange(len(index)), breaks):
            if len(run) > 1:
                segments.append(Segment(plane_index, x[run], y[run]))
    return segments


@dataclass(frozen=True)
class TrackCorrection:
    """ what the corrected drawing of a track changes from what was stored """
    t0: float               # ns, nan without a T0
    tick_shift: float       # ticks added to the drift time
    tpcs: np.ndarray        # (cryostat, tpc) per trajectory point as drawn
    moved: float            # fraction of valid points drawn in another TPC than their position's

    @property
    def corrected(self):
        return self.tick_shift != 0 or self.moved > 0

    def describe(self):
        parts = []
        if not np.isnan(self.t0):
            parts.append(f'T0 {self.t0 / 1000:.1f} us: drawn {self.tick_shift:+.0f} ticks')
        if self.moved > 0:
            parts.append(f'{100 * self.moved:.0f}% of points drawn in their hits\' TPC')
        return '; '.join(parts) or 'drawn as stored'


def track_correction(layout, tracks, row, t0=np.nan, track_hits=None, hits=None) -> TrackCorrection:
    """
    Corrections for drawing a track on its hits: undo its T0 (ns, from
    track_t0s) and draw each point in the TPC of its hits, or the TPC whose
    readout window contains it without associations
    """
    xyz = tracks.points(row, valid_only=False)
    valid = tracks.valid[tracks.point_offset[row]:tracks.point_offset[row + 1]]
    shift = 0. if np.isnan(t0) else t0 * ticks_per_ns(layout.geom)
    tpcs = None
    if track_hits is not None and hits is not None:
        mine = track_hits.left == row
        tpcs = tpcs_from_hits(len(xyz), track_hits.meta_index[mine], track_hits.right[mine], hits)
        if np.all(tpcs[:, 0] < 0):
            tpcs = None
    if tpcs is None:
        tpcs = readout_tpcs(layout.geom.detector, xyz, layout.n_ticks, shift)
    tpcs[~valid] = -1
    by_position = position_tpcs(layout.geom.detector, xyz)
    moved = np.mean(np.any(tpcs[valid] != by_position[valid], axis=1)) if valid.any() else 0.
    return TrackCorrection(t0=t0, tick_shift=shift, tpcs=tpcs, moved=float(moved))


def object_t0s(n_objects, pfp_objects=None, pfp_t0s=None, t0s=None):
    """
    T0 (ns) per object (track, vertex, spacepoint...) via its PFParticle:
    pfp_objects associates PFParticles (left) with the objects (right),
    pfp_t0s PFParticles with anab::T0s. nan without one.
    """
    result = np.full(n_objects, np.nan)
    if pfp_objects is None or pfp_t0s is None or t0s is None:
        return result
    t0_of_pfp = dict(zip(pfp_t0s.left, t0s.time[pfp_t0s.right]))
    for pfp, obj in zip(pfp_objects.left, pfp_objects.right):
        if pfp in t0_of_pfp and 0 <= obj < n_objects:
            result[obj] = t0_of_pfp[pfp]
    return result


def track_t0s(n_tracks, pfp_tracks=None, pfp_t0s=None, t0s=None):
    """ T0 (ns) per track, see object_t0s """
    return object_t0s(n_tracks, pfp_tracks, pfp_t0s, t0s)


def object_hit_tpcs(n_objects, object_hits=None, hits=None):
    """
    (cryostat, tpc) per object from the hits associated with it (e.g. a
    spacepoint's hits, all in one TPC); (-1, -1) for objects without hits
    """
    tpcs = np.full((n_objects, 2), -1, dtype=int)
    if object_hits is None or hits is None or len(object_hits) == 0:
        return tpcs
    ok = (object_hits.left >= 0) & (object_hits.left < n_objects)
    tpcs[object_hits.left[ok], 0] = hits.cryostat[object_hits.right[ok]]
    tpcs[object_hits.left[ok], 1] = hits.tpc[object_hits.right[ok]]
    return tpcs


@dataclass(frozen=True)
class PointCorrection:
    """ what the corrected drawing of point-like objects (vertices, spacepoints) changes """
    t0: np.ndarray          # ns per point, nan without a T0
    tick_shift: np.ndarray  # ticks added to the drift time, per point
    tpcs: np.ndarray        # (cryostat, tpc) per point as drawn
    moved: np.ndarray       # per point: shifted, or drawn in another TPC than its position's

    def describe(self, i):
        parts = []
        if not np.isnan(self.t0[i]):
            parts.append(f'T0 {self.t0[i] / 1000:.1f} us: drawn {self.tick_shift[i]:+.0f} ticks')
        if self.moved[i] and np.isnan(self.t0[i]):
            parts.append("drawn in its hits' TPC")
        return '; '.join(parts) or 'drawn as stored'


def point_correction(layout, points, t0=None, hit_tpcs=None) -> PointCorrection:
    """
    Corrections for drawing points on their hits: undo their T0 (ns per
    point, see object_t0s) and draw each in the TPC of its hits (per point,
    see object_hit_tpcs), else the TPC whose readout window contains it
    """
    xyz = points.xyz
    n = len(xyz)
    t0 = np.full(n, np.nan) if t0 is None else np.asarray(t0, dtype=float)
    shift = np.nan_to_num(t0) * ticks_per_ns(layout.geom)
    tpcs = readout_tpcs(layout.geom.detector, xyz, layout.n_ticks, shift)
    if hit_tpcs is not None:
        known = np.asarray(hit_tpcs)[:, 0] >= 0
        tpcs[known] = np.asarray(hit_tpcs)[known]
    moved = (shift != 0) | np.any(tpcs != position_tpcs(layout.geom.detector, xyz), axis=1)
    return PointCorrection(t0=t0, tick_shift=shift, tpcs=tpcs, moved=moved)


def project_objects(layout, points, correction=None):
    """
    Point-like objects in a view: {plane index: (indices, x, y)}, as stored
    (by position, no shift) without a correction, else as it says
    """
    if correction is None:
        return project_points(layout, points.xyz, 0., position_tpcs(layout.geom.detector, points.xyz))
    return project_points(layout, points.xyz, correction.tick_shift, correction.tpcs)


def track_segments(layout, tracks, row, correction=None) -> List[Segment]:
    """
    View segments of a track's valid trajectory points: as stored (by
    position, no T0 shift) without a correction, else as the correction
    says (see track_correction)
    """
    xyz = tracks.points(row, valid_only=False)
    valid = tracks.valid[tracks.point_offset[row]:tracks.point_offset[row + 1]]
    if correction is None:
        tpcs, shift = position_tpcs(layout.geom.detector, xyz), 0.
    else:
        tpcs, shift = correction.tpcs.copy(), correction.tick_shift
    tpcs[~valid] = -1
    return project_path(layout, xyz, shift, tpcs)
