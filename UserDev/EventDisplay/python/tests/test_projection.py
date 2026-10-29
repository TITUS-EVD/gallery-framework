"""
Projection of 3D points into TPC views (titus.display.projection), against
the detector description and against the hits of real Pandora tracks
(tests/data/sbnd-reco2-pandora: events 0 and 2 of an SBND reco2 file with
gaushit, pandoraTrack and Pandora's vertices, spacepoints, T0s and their
associations, made with tools/trim_dump.py).
"""

import pathlib

import numpy as np
import pytest

from titus.display import (ViewLayout, hit_positions, object_hit_tpcs, object_t0s, particle_tpcs,
                           point_correction, position_tpcs, project_objects, project_path, project_points,
                           readout_tpcs, ticks_per_ns, tpcs_from_hits, track_correction, track_segments,
                           track_t0s)
from titus.geometry import Geometry
from titus.readers import DumpReader

DUMP = pathlib.Path(__file__).parent / 'data' / 'sbnd-reco2-pandora'


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
    assert any(c.describe() == "corrected: nothing to change (no T0, already in their hits' TPC)" for c in untagged)


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


def test_point_correction(sbnd):
    from titus.readers import Points
    layout = ViewLayout(sbnd, plane=2)
    # in TPC 0; past the cathode (in TPC 1's volume) but read out by TPC 0; behind TPC 1's anode
    points = Points(id=[0, 1, 2], x=[-100., 10., 220.], y=[0., 0., 0.], z=[250., 250., 250.])
    correction = point_correction(layout, points, t0=[-500000., np.nan, np.nan],
                                  hit_tpcs=[[-1, -1], [0, 0], [-1, -1]])
    assert correction.tick_shift.tolist() == pytest.approx([-1000., 0., 0.])
    assert correction.tpcs.tolist() == [[0, 0], [0, 0], [0, 1]]
    assert correction.moved.tolist() == [True, True, False]
    assert correction.describe(0) == 'T0 -500.0 us: drawn -1000 ticks'
    assert correction.describe(1) == "drawn in its hits' TPC"
    assert correction.describe(2) == "corrected: nothing to change (no T0, already in its hits' TPC)"

    stored = project_objects(layout, points)
    corrected = project_objects(layout, points, correction)
    # as stored, the point past the cathode is in TPC 1 (plane index 5)
    assert sorted(stored[5][0].tolist()) == [1, 2] and stored[2][0].tolist() == [0]
    assert sorted(corrected[2][0].tolist()) == [0, 1] and corrected[5][0].tolist() == [2]
    # the T0 shift, in the unflipped TPC 0 block
    assert corrected[2][2][0] - stored[2][2][0] == pytest.approx(-1000.)


def test_object_hit_tpcs():
    from titus.readers import Associations, Hits
    hits = Hits(channel=[0, 0], cryostat=[0, 0], tpc=[1, 0], plane=[0, 0], wire=[0, 0], peak_time=[0, 0],
                rms=[1, 1], integral=[1, 1], peak_amplitude=[1, 1], start_tick=[0, 0], end_tick=[0, 0])
    assns = Associations(left=[0, 0, 2], right=[0, 0, 1], left_product=[0, 0, 0], right_product=[1, 1, 1],
                         meta_index=[-1, -1, -1], product_tags=['pandora::Reco2', 'gaushit::Reco1'])
    assert object_hit_tpcs(3, assns, hits).tolist() == [[0, 1], [-1, -1], [0, 0]]
    assert (object_hit_tpcs(2) == -1).all()


def spacepoint_residuals(geom, dump, entry, corrected=True):
    """ view-coordinate residuals (hit - projected spacepoint) of each spacepoint's hit """
    hits = dump.hits(entry, 'gaushit')
    points = dump.get(entry, 'spacepoints', 'pandora')
    spacepoint_hits = dump.get(entry, 'spacepoint_hits', 'pandora')
    t0 = object_t0s(len(points), dump.get(entry, 'pfp_spacepoints', 'pandora'),
                    dump.get(entry, 'pfp_t0s', 'pandora'), dump.get(entry, 't0s', 'pandora'))
    hit_tpcs = object_hit_tpcs(len(points), spacepoint_hits, hits)
    residuals = []
    for plane in range(geom.nPlanes()):
        layout = ViewLayout(geom, plane)
        positions = hit_positions(layout, hits)
        hit_x, hit_y = dict(zip(positions.index, positions.x)), dict(zip(positions.index, positions.y))
        correction = point_correction(layout, points, t0, hit_tpcs) if corrected else None
        for block, (index, x, y) in project_objects(layout, points, correction).items():
            where = dict(zip(index, range(len(index))))
            on_block = hits.on_plane(*geom.detector.planes[block].id)
            for s, h in zip(spacepoint_hits.left, spacepoint_hits.right):
                if s in where and on_block[h] and h in hit_x:
                    residuals.append((hit_x[h] - x[where[s]], hit_y[h] - y[where[s]]))
    return np.array(residuals), len(spacepoint_hits)


@pytest.mark.parametrize('entry', [0, 1])
def test_spacepoints_lie_on_their_hits(sbnd, dump, entry):
    """
    Corrected (T0 undone, TPC of their hit), every Pandora spacepoint is drawn
    on its hit; as stored, out-of-time and T0-tagged ones aren't
    """
    corrected, n_hits = spacepoint_residuals(sbnd, dump, entry)
    assert len(corrected) == n_hits
    assert np.mean(np.abs(corrected[:, 0]) < 1) > 0.9
    assert np.mean(np.abs(corrected[:, 1]) < 3) > 0.98
    stored, _ = spacepoint_residuals(sbnd, dump, entry, corrected=False)
    # as stored, some aren't even drawn in their hit's TPC, and some are off by a T0
    assert len(stored) < 0.97 * n_hits
    assert np.mean(np.abs(stored[:, 1]) < 3) < 0.8


def vertex_distances(geom, dump, entry, corrected=True):
    """
    Per vertex and view: distance (cm, in the view) to the nearest hit of its
    own PFParticle, for vertices whose particle has hits in the view
    """
    hits = dump.hits(entry, 'gaushit')
    vertices = dump.get(entry, 'vertices', 'pandora')
    points = dump.get(entry, 'spacepoints', 'pandora')
    pfp_vertices = dump.get(entry, 'pfp_vertices', 'pandora')
    pfp_spacepoints = dump.get(entry, 'pfp_spacepoints', 'pandora')
    spacepoint_hits = dump.get(entry, 'spacepoint_hits', 'pandora')
    t0 = object_t0s(len(vertices), pfp_vertices, dump.get(entry, 'pfp_t0s', 'pandora'), dump.get(entry, 't0s', 'pandora'))
    tpcs = particle_tpcs(vertices.xyz, pfp_vertices, pfp_spacepoints, points,
                         object_hit_tpcs(len(points), spacepoint_hits, hits))
    hit_of_point = dict(zip(spacepoint_hits.left, spacepoint_hits.right))
    hits_of_pfp = {}
    for pfp, point in zip(pfp_spacepoints.left, pfp_spacepoints.right):
        if point in hit_of_point:
            hits_of_pfp.setdefault(pfp, []).append(hit_of_point[point])
    pfp_of_vertex = dict(zip(pfp_vertices.right, pfp_vertices.left))
    distances = []
    for plane in range(geom.nPlanes()):
        layout = ViewLayout(geom, plane)
        positions = hit_positions(layout, hits)
        hit_x, hit_y = dict(zip(positions.index, positions.x)), dict(zip(positions.index, positions.y))
        correction = point_correction(layout, vertices, t0, tpcs) if corrected else None
        for _, (index, x, y) in project_objects(layout, vertices, correction).items():
            for i, vx, vy in zip(index, x, y):
                own = [h for h in hits_of_pfp.get(pfp_of_vertex.get(i), []) if h in hit_x]
                if len(own) >= 5:
                    distances.append(min(np.hypot((hit_x[h] - vx) * geom.wire2cm(), (hit_y[h] - vy) * geom.time2cm())
                                         for h in own))
    return np.array(distances)


@pytest.mark.parametrize('entry', [0, 1])
def test_vertices_lie_on_their_particles(sbnd, dump, entry):
    """
    Corrected (T0 undone, TPC of their particle's nearest spacepoint),
    Pandora vertices sit on their own particle's hits
    """
    corrected = vertex_distances(sbnd, dump, entry)
    assert len(corrected) >= 40
    assert np.median(corrected) < 0.5
    assert np.mean(corrected < 2) > 0.85
    assert np.mean(corrected > 20) < 0.05
    stored = vertex_distances(sbnd, dump, entry, corrected=False)
    assert np.mean(stored > 20) > 0.3


def test_cone_points():
    from titus.display import cone_points
    xyz = cone_points([1., 2., 3.], [0., 0., 2.], length=10., half_angle=np.arctan(0.5), n_circle=8)
    assert xyz[0].tolist() == [1, 2, 3]
    end = xyz[1:]
    assert np.allclose(end[:, 2], 13.)
    assert np.allclose(np.hypot(end[:, 0] - 1, end[:, 1] - 2), 5.)


def test_convex_hull():
    from titus.display import convex_hull
    x = np.array([0., 2., 2., 0., 1., 1.])
    y = np.array([0., 0., 2., 2., 1., 0.5])
    assert sorted(convex_hull(x, y)) == [0, 1, 2, 3]
    assert convex_hull(np.array([0., 1.]), np.array([0., 1.])) == [0, 1]


@pytest.mark.parametrize('x, block', [(-100., 2), (100., 5)])
def test_project_cone(sbnd, x, block):
    """ a cone along z: a triangle from the apex wire to the end wire, symmetric in ticks """
    from titus.display import project_cone
    layout = ViewLayout(sbnd, plane=2)
    (polygon,) = project_cone(layout, [x, 0., 100.], [0., 0., 1.], length=100., half_angle=0.1)
    assert polygon.plane_index == block
    plane = sbnd.detector.planes[block]
    apex_x = layout.wire_to_view(block, plane.wire_coordinate([x, 0., 100.]))
    end_x = layout.wire_to_view(block, plane.wire_coordinate([x, 0., 200.]))
    assert len(polygon.x) == 3
    assert sorted(polygon.x) == pytest.approx(sorted([apex_x, end_x, end_x]))
    apex_y = layout.tick_to_view(block, plane.x_to_ticks(x))
    spread = 100 * np.tan(0.1) / sbnd.time2cm()  # ticks, either side of the axis
    ends = sorted(polygon.y[np.isclose(polygon.x, end_x)])
    assert ends == pytest.approx([apex_y - spread, apex_y + spread], rel=1e-3)


def test_united_track_segments(sbnd):
    """
    points in ticks cut by "unite cathodes" (past the cathode) are left out:
    a late cosmic read out by TPC 0 but placed past the cathode, x > 0
    """
    full = ViewLayout(sbnd, plane=2)
    layout = ViewLayout(sbnd, plane=2, removed_ticks=300)
    path = np.stack([np.linspace(-100, 30, 40), np.zeros(40), np.full(40, 250.)], axis=1)
    tpcs = np.tile([0, 0], (40, 1))
    (segment,) = project_path(layout, path, tpcs=tpcs)
    assert segment.y.max() <= layout.n_ticks - 300
    (whole,) = project_path(full, path, tpcs=tpcs)
    assert len(segment.y) < len(whole.y)


# --- showers, with their particles' spacepoints (sbnd-reco2-step3) ---

STEP3 = pathlib.Path(__file__).parent / 'data' / 'sbnd-reco2-step3'


def shower_spacepoints(reader, tag='pandoraShower'):
    ''' per shower with >= 20 spacepoints: (start, unit direction, open angle, its particle's spacepoints) '''
    showers = reader.get(0, 'showers', tag)
    pfp_showers = reader.get(0, 'pfp_showers', tag)
    pfp_spacepoints = reader.get(0, 'pfp_spacepoints', pfp_showers.product_tags[pfp_showers.left_product[0]])
    xyz = reader.get(0, 'spacepoints', pfp_spacepoints.product_tags[pfp_spacepoints.right_product[0]]).xyz
    out = []
    for pfp, row in zip(pfp_showers.left, pfp_showers.right):
        points = xyz[pfp_spacepoints.right[pfp_spacepoints.left == pfp]]
        if len(points) < 20 or not showers.length[row] > 0:
            continue
        start = np.array([showers.start_x[row], showers.start_y[row], showers.start_z[row]])
        direction = np.array([showers.dir_x[row], showers.dir_y[row], showers.dir_z[row]])
        out.append((start, direction / np.linalg.norm(direction), showers.open_angle[row], points))
    return out


def test_shower_open_angle_is_a_half_angle():
    ''' the drawn cone holds the core of its particle's spacepoints (OPEN_ANGLE_TO_HALF_ANGLE) '''
    from titus.display.projection import OPEN_ANGLE_TO_HALF_ANGLE
    inside = {0.5: [], 1.0: []}
    for start, direction, open_angle, points in shower_spacepoints(DumpReader(STEP3)):
        if np.degrees(open_angle) < 5:
            continue  # track-like: spacepoint scatter dominates any cone this thin
        rel = points - start
        distance = np.linalg.norm(rel, axis=1)
        ahead = (distance > 1) & (rel @ direction > 0)
        angle = np.arccos(np.clip(rel[ahead] @ direction / distance[ahead], -1, 1))
        for factor in inside:
            inside[factor].append(np.mean(angle <= factor * open_angle))
    assert len(inside[1.0]) >= 3
    assert OPEN_ANGLE_TO_HALF_ANGLE == 1.0
    assert 0.45 < np.median(inside[1.0]) < 0.8
    assert np.median(inside[0.5]) < 0.35


def test_shower_starts_on_their_particles_spacepoints():
    showers = shower_spacepoints(DumpReader(STEP3))
    assert showers
    nearest = [np.linalg.norm(points - start, axis=1).min() for start, _, _, points in showers]
    assert max(nearest) < 1.0
