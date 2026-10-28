"""
Projection of 3D points into TPC views (titus.display.projection), against
the detector description and against the hits of real Pandora tracks
(tests/data/sbnd-reco2-tracks: event 0 of an SBND reco2 file, gaushit and
pandoraTrack only, made with tools/trim_dump.py).
"""

import pathlib

import numpy as np
import pytest

from titus.display import ViewLayout, hit_positions, project_path, project_points, ticks_per_ns
from titus.geometry import Geometry
from titus.readers import DumpReader

DUMP = pathlib.Path(__file__).parent / 'data' / 'sbnd-reco2-tracks'


@pytest.fixture(scope='module')
def sbnd():
    return Geometry.load('sbnd')


@pytest.fixture(scope='module')
def dump():
    return DumpReader(DUMP)


def test_points_land_on_layout(sbnd):
    rng = np.random.default_rng(3)
    for plane in range(3):
        layout = ViewLayout(sbnd, plane)
        for block in layout.blocks:
            geo = sbnd.detector.planes[block.plane_index]
            tpc = sbnd.detector.tpc(geo.id.cryostat, geo.id.tpc)
            points = rng.uniform(tpc.active_bounds.min, tpc.active_bounds.max, size=(50, 3))
            index, x, y = project_points(layout, points)[block.plane_index]
            assert len(index) == 50
            np.testing.assert_allclose(x, layout.wire_to_view(block.plane_index, geo.wire_coordinate(points)))
            np.testing.assert_allclose(y, layout.tick_to_view(block.plane_index, geo.x_to_ticks(points[:, 0])))
            # inside the block (the active volume reaches up to ~11 wire pitches past the end wires)
            assert np.all((x >= block.wire_offset - 15) & (x <= block.wire_offset + block.n_wires + 15))
            assert np.all((y >= block.time_offset) & (y <= block.time_offset + layout.n_ticks))


def test_path_split_at_cathode(sbnd):
    """ a straight line through the cathode: one segment per TPC """
    path = np.stack([np.linspace(-150, 150, 31), np.zeros(31), np.full(31, 250.)], axis=1)
    layout = ViewLayout(sbnd, plane=2)
    segments = project_path(layout, path)
    assert sorted(s.plane_index for s in segments) == [2, 5]
    # TPC bounds are inclusive: the point at the cathode (x = 0) goes to TPC 0
    assert sorted(len(s.x) for s in segments) == [15, 16]
    # points outside every TPC are dropped
    assert project_path(layout, [[0, 500, 0], [0, 600, 0]]) == []


def test_tick_shift(sbnd):
    layout = ViewLayout(sbnd, plane=0)
    point = [[-100., 0., 250.]]
    _, _, y0 = project_points(layout, point)[0]
    _, _, y1 = project_points(layout, point, tick_shift=10.)[0]
    assert y1 - y0 == pytest.approx(10.)
    # flipped TPC: later times are drawn upwards from the far end
    _, _, y0 = project_points(layout, [[100., 0., 250.]])[4]
    _, _, y1 = project_points(layout, [[100., 0., 250.]], tick_shift=10.)[4]
    assert y1 - y0 == pytest.approx(-10.)
    assert ticks_per_ns(sbnd) == pytest.approx(1 / 500.)


def track_residuals(layout, tracks, hits, row, tick_shift=0.):
    """ per block, view-y residual (ticks) of each projected point to the nearest hit on its wire """
    positions = hit_positions(layout, hits)
    planes = layout.geom.detector.planes
    result = {}
    for segment in project_path(layout, tracks.points(row), tick_shift):
        on_block = np.isin(positions.index, np.flatnonzero(hits.on_plane(*planes[segment.plane_index].id)))
        hx, hy = positions.x[on_block], positions.y[on_block]
        residuals = []
        for x, y in zip(segment.x, segment.y):
            near = np.abs(hx - x) < 0.6
            if near.any():
                d = hy[near] - y
                residuals.append(d[np.argmin(np.abs(d))])
        result.setdefault(segment.plane_index, []).extend(residuals)
    return {k: np.array(v) for k, v in result.items()}


def test_tracks_lie_on_their_hits(sbnd, dump):
    """
    Pandora tracks without a T0 shift lie on their hits: median residual to
    the nearest hit within a tick in all three views of a TPC, in both TPCs
    (the second flipped). Without the track-hit associations the nearest hit
    is often another track's, so only the median is required. Tracks with a
    T0 are shifted in x by Pandora (by ~1735 ticks in this event) and don't,
    until the T0 is undone.
    """
    tracks = dump.get(0, 'tracks', 'pandoraTrack')
    hits = dump.hits(0, 'gaushit')
    medians = {}
    for plane in range(3):
        layout = ViewLayout(sbnd, plane)
        for row in range(len(tracks)):
            for plane_index, r in track_residuals(layout, tracks, hits, row).items():
                if len(r) >= 10:
                    medians[(row, plane_index)] = np.median(r)
    on_tpc = {(row, tpc) for row in range(len(tracks)) for tpc in (0, 1)
              if all(abs(medians.get((row, 3 * tpc + p), np.inf)) < 1 for p in range(3))}
    assert len(on_tpc) >= 8
    assert {tpc for _, tpc in on_tpc} == {0, 1}
    # the T0-shifted group
    shifted = [m for m in medians.values() if abs(abs(m) - 1735) < 15]
    assert len(shifted) >= 10
