"""
Projection of 3D points into TPC views (titus.display.projection), against
the detector description and against the hits of real Pandora tracks
(tests/data/sbnd-reco2-tracks: events 0 and 2 of an SBND reco2 file with
gaushit, pandoraTrack, their associations and Pandora's T0s, made with
tools/trim_dump.py).
"""

import pathlib

import numpy as np
import pytest

from titus.display import (ViewLayout, hit_positions, position_tpcs, project_path, project_points,
                           readout_tpcs, ticks_per_ns, tpcs_from_hits, track_correction, track_segments,
                           track_t0s)
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


def associated_residuals(geom, dump, entry, use_t0=True, use_hit_tpcs=True):
    """
    Per track: view-coordinate residuals (hit - projected trajectory point)
    of each associated hit against its own trajectory point (TrackHitMeta),
    over all views
    """
    hits = dump.hits(entry, 'gaushit')
    tracks = dump.get(entry, 'tracks', 'pandoraTrack')
    track_hits = dump.get(entry, 'track_hits', 'pandoraTrack')
    t0s = track_t0s(len(tracks), dump.get(entry, 'pfp_tracks', 'pandoraTrack'),
                    dump.get(entry, 'pfp_t0s', 'pandora'), dump.get(entry, 't0s', 'pandora'))
    shifts = np.nan_to_num(t0s) * ticks_per_ns(geom) if use_t0 else np.zeros(len(tracks))
    residuals = {}
    for plane in range(geom.nPlanes()):
        layout = ViewLayout(geom, plane)
        positions = hit_positions(layout, hits)
        hit_x = dict(zip(positions.index, positions.x))
        hit_y = dict(zip(positions.index, positions.y))
        for row in range(len(tracks)):
            mine = track_hits.left == row
            hit_rows, point_index = track_hits.right[mine], track_hits.meta_index[mine]
            xyz = tracks.points(row, valid_only=False)
            tpcs = tpcs_from_hits(len(xyz), point_index, hit_rows, hits) if use_hit_tpcs else None
            projected = project_points(layout, xyz, shifts[row], tpcs)
            for block_index, (index, x, y) in projected.items():
                where = dict(zip(index, range(len(index))))
                on_block = hits.on_plane(*geom.detector.planes[block_index].id)
                for h, m in zip(hit_rows, point_index):
                    if on_block[h] and m in where and h in hit_x:
                        residuals.setdefault(row, []).append((hit_x[h] - x[where[m]], hit_y[h] - y[where[m]]))
    return {row: np.array(r) for row, r in residuals.items()}


@pytest.mark.parametrize('entry', [0, 1])
def test_tracks_lie_on_their_hits(sbnd, dump, entry):
    """
    Every Pandora track lies on its own hits in all views and both TPCs,
    once its T0 shift is undone and each point is drawn in its hits' TPC
    """
    residuals = associated_residuals(sbnd, dump, entry)
    long_tracks = {row: r for row, r in residuals.items() if len(r) >= 30}
    assert len(long_tracks) >= 10
    medians = np.array([np.median(r, axis=0) for r in long_tracks.values()])
    # a wrong T0 or TPC moves a track by hundreds of ticks
    assert np.all(np.abs(medians) < 10)
    # within a wire and a tick; the few exceptions are Pandora's fit, not the projection
    assert np.mean(np.all(np.abs(medians) < 1, axis=1)) >= 0.9
    every_hit = np.concatenate(list(residuals.values()))
    assert np.mean(np.abs(every_hit[:, 0]) < 1) > 0.9
    assert np.mean(np.abs(every_hit[:, 1]) < 3) > 0.95


def test_t0_shift_is_needed(sbnd, dump):
    """ event 2 (entry 1) has four T0-tagged cathode crossers: off by hundreds of ticks without it """
    with_t0 = associated_residuals(sbnd, dump, 1)
    without = associated_residuals(sbnd, dump, 1, use_t0=False)
    shifted = [row for row, r in without.items() if len(r) >= 30 and abs(np.median(r[:, 1])) > 100]
    assert len(shifted) >= 4
    for row in shifted:
        assert abs(np.median(with_t0[row][:, 1])) < 1


def test_hit_tpcs_are_needed(sbnd, dump):
    """
    Out-of-time cosmics without a T0 are placed behind the anode or past the
    cathode: assigning their points by position puts them in the wrong TPC
    """
    tracks = dump.get(0, 'tracks', 'pandoraTrack')
    track_hits = dump.get(0, 'track_hits', 'pandoraTrack')
    hits = dump.hits(0, 'gaushit')
    wrong = 0
    for row in range(len(tracks)):
        mine = track_hits.left == row
        xyz = tracks.points(row, valid_only=False)
        by_hits = tpcs_from_hits(len(xyz), track_hits.meta_index[mine], track_hits.right[mine], hits)
        by_position = sbnd.detector.tpc_of(xyz)
        wrong += np.sum(np.any(by_hits != by_position, axis=1))
    assert wrong > 1000
    # by position (readout_tpcs), ~30% of the hits in this event lose their
    # trajectory point: it is drawn in the other TPC's block, or not at all
    n_pairs = sum(len(r) for r in associated_residuals(sbnd, dump, 0).values())
    n_by_position = sum(len(r) for r in associated_residuals(sbnd, dump, 0, use_hit_tpcs=False).values())
    assert n_by_position < 0.8 * n_pairs


def test_tpcs_from_hits_fills_gaps():
    from titus.readers import Hits
    hits = Hits(channel=[0, 0], cryostat=[0, 0], tpc=[0, 1], plane=[2, 2], wire=[1, 2], peak_time=[0, 0],
                rms=[1, 1], integral=[1, 1], peak_amplitude=[1, 1], start_tick=[0, 0], end_tick=[0, 0])
    tpcs = tpcs_from_hits(6, point_index=[1, 4], hit_rows=[0, 1], hits=hits)
    assert tpcs[:, 1].tolist() == [0, 0, 0, 1, 1, 1]
    assert (tpcs_from_hits(3, [], [], hits) == -1).all()


def test_readout_tpcs_behind_anode(sbnd):
    """ an early cosmic placed behind the TPC 1 anode (x > 202 cm) is still read out by TPC 1 """
    tpcs = readout_tpcs(sbnd.detector, [[220., 0., 250.], [100., 0., 250.], [260., 0., 250.]], 3400)
    assert tpcs.tolist() == [[0, 1], [0, 1], [-1, -1]]


def reco2_event(dump, entry):
    return dict(tracks=dump.get(entry, 'tracks', 'pandoraTrack'),
                track_hits=dump.get(entry, 'track_hits', 'pandoraTrack'),
                hits=dump.hits(entry, 'gaushit'),
                t0s=track_t0s(len(dump.get(entry, 'tracks', 'pandoraTrack')),
                              dump.get(entry, 'pfp_tracks', 'pandoraTrack'),
                              dump.get(entry, 'pfp_t0s', 'pandora'), dump.get(entry, 't0s', 'pandora')))


def segment_points(segments):
    return sum(len(s.x) for s in segments)


def test_corrected_and_as_stored(sbnd, dump):
    """
    As stored, a T0-tagged track is drawn where Pandora put it, off its hits;
    corrected, the T0 is undone and the correction says so
    """
    ev = reco2_event(dump, 1)
    layout = ViewLayout(sbnd, plane=2)
    tagged = np.flatnonzero(~np.isnan(ev['t0s']))
    assert len(tagged) >= 4
    for row in tagged[:4]:
        correction = track_correction(layout, ev['tracks'], row, ev['t0s'][row], ev['track_hits'], ev['hits'])
        assert correction.corrected and correction.tick_shift == pytest.approx(ev['t0s'][row] / 500.)
        assert correction.describe().startswith(f"T0 {ev['t0s'][row] / 1000:.1f} us")
        stored = track_segments(layout, ev['tracks'], row)
        corrected = track_segments(layout, ev['tracks'], row, correction)
        y_stored = np.concatenate([s.y for s in stored if s.plane_index == 2])
        y_corrected = np.concatenate([s.y for s in corrected if s.plane_index == 2])
        # unflipped TPC 0: shifted by the T0 in ticks
        assert np.median(y_corrected) - np.median(y_stored) == pytest.approx(correction.tick_shift, abs=1)


def test_untagged_track_needs_no_shift_but_may_move(sbnd, dump):
    ev = reco2_event(dump, 0)
    layout = ViewLayout(sbnd, plane=2)
    corrections = [track_correction(layout, ev['tracks'], row, ev['t0s'][row], ev['track_hits'], ev['hits'])
                   for row in range(len(ev['tracks']))]
    untagged = [c for c in corrections if np.isnan(c.t0)]
    assert all(c.tick_shift == 0 for c in untagged)
    # out-of-time cosmics: some points drawn in their hits' TPC
    moved = [c for c in untagged if c.moved > 0]
    assert moved and all('in their hits' in c.describe() for c in moved)
    assert any(c.describe() == 'drawn as stored' for c in untagged)


def test_as_stored_draws_every_valid_point(sbnd, dump):
    """ as stored, nothing is dropped: points outside every TPC go to the nearest """
    ev = reco2_event(dump, 0)
    for row in range(len(ev['tracks'])):
        n_valid = int(ev['tracks'].valid[ev['tracks'].point_offset[row]:ev['tracks'].point_offset[row + 1]].sum())
        drawn = sum(segment_points(track_segments(ViewLayout(sbnd, plane=p), ev['tracks'], row)) for p in range(3))
        # each point in one block of each of the 3 views (single points between TPC changes aren't segments)
        assert drawn >= 3 * n_valid - 3 * 10


def test_position_tpcs_nearest(sbnd):
    tpcs = position_tpcs(sbnd.detector, [[-100., 0., 250.], [230., 0., 250.], [-230., 0., 250.]])
    assert tpcs.tolist() == [[0, 0], [0, 1], [0, 0]]
