"""
Space charge (titus.display.space_charge): the shipped SBND map, its
interpolation, which products are in true space, and that it draws MC and
Pandora's SCE-corrected objects on their charge (an SBND reco2 event).
"""

import pathlib
from unittest import mock
import sys

import numpy as np
import pytest

sys.modules.setdefault('ROOT', mock.MagicMock())

from titus.display import ViewLayout, hit_positions, mc_t0, track_correction, track_segments  # noqa: E402
from titus.display.space_charge import (SpaceChargeMap, available_sce_maps, displaced, in_true_space,  # noqa: E402
                                        load_sce_map)
from titus.drawables.mctrack import mctracks_as_tracks  # noqa: E402
from titus.geometry import Geometry  # noqa: E402
from titus.readers import DumpReader, Hits, Points, Showers  # noqa: E402

DATA = pathlib.Path(__file__).parent / 'data'
SBND_MAP = 'SCEoffsets_SBND_E500_dualmap_CV_voxelTH3'


@pytest.fixture(scope='module')
def sbnd_map():
    return load_sce_map(SBND_MAP)


@pytest.fixture(scope='module')
def event():
    return DumpReader(DATA / 'sbnd-reco2-sce')


def small_map():
    ''' 2x2x2 voxels of 1 cm over [0, 2) per axis, a dual map: east dx = 1, west dx = x index '''
    axes = [[2, 0., 2.]] * 3
    zero = np.zeros((2, 2, 2))
    west_dx = np.zeros((2, 2, 2))
    west_dx[1] = 2.
    return SpaceChargeMap('small', {'E': [zero + 1, zero, zero], 'W': [west_dx, zero, zero + 0.5]}, axes)


def test_the_shipped_sbnd_map(sbnd_map):
    assert SBND_MAP in available_sce_maps()
    assert sbnd_map.name == SBND_MAP and sbnd_map.meta['sides'] == {'E': 'x < 0', 'W': 'x >= 0'}
    assert load_sce_map('SCEoffsets_unknown') is None
    # displacements of a centimetre or so, inside the detector
    d = sbnd_map.displacement(np.array([[-100., 150., 250.], [100., -150., 50.]]))
    assert np.all(np.abs(d) < 5) and np.any(np.abs(d) > 0.1)


def test_map_files_written_without_numpy_load(tmp_path):
    ''' tools/dump_sce_map.py writes .npz by hand (PyROOT only): the voxels in (x, y, z) order '''
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'tools'))
    import dump_sce_map
    axes = [[2, 0., 2.], [3, -3., 3.], [4, 0., 8.]]
    values = {f'd{a}': [100 * i + 10 * j + k + 0.25 * n for i in range(2) for j in range(3) for k in range(4)]
              for n, a in enumerate('xyz')}
    path = tmp_path / 'small.npz'
    dump_sce_map.write_map(path, {'all': values}, axes, {'name': 'small', 'sides': {'all': 'everywhere'}})
    m = SpaceChargeMap.load(path)
    assert m.name == 'small'
    # voxel (1, 2, 3) is at (1.5, 2, 7)
    np.testing.assert_allclose(m.displacement(np.array([[1.5, 2., 7.]])), [[123., 123.25, 123.5]])


def test_interpolation_between_voxel_centers():
    m = small_map()
    # at the voxel centers (0.5, 1.5): the stored values; halfway between, their mean
    points = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.0, 0.5, 0.5], [1.5, 1.5, 1.5]])
    np.testing.assert_allclose(m.displacement(points)[:, 0], [0., 2., 1., 2.])
    np.testing.assert_allclose(m.displacement(points)[:, 2], 0.5)
    # beyond the outermost centers: the edge voxels' values
    np.testing.assert_allclose(m.displacement(np.array([[5., -3., 0.]]))[:, 0], [2.])
    # x < 0: the east map
    np.testing.assert_allclose(m.displacement(np.array([[-0.5, 0.5, 0.5]])), [[1., 0., 0.]])
    np.testing.assert_allclose(m.displace(np.array([[1.5, 0.5, 0.5]])), [[3.5, 0.5, 1.0]])


def test_which_products_are_in_true_space():
    assert in_true_space('mctracks', 'mcreco::G4') and in_true_space('mctruths', 'generator::GenieGen')
    for kind in ('tracks', 'showers', 'vertices', 'spacepoints'):
        assert in_true_space(kind, 'pandoraSCE::Reco2') and in_true_space(kind, 'pandoraSCEShowerSBN::Reco2')
        assert not in_true_space(kind, 'pandora::Reco2') and not in_true_space(kind, 'pandoraShower::Reco2')
    # what is drawn where the charge was read out
    assert not in_true_space('hits', 'pandoraSCE::Reco2') and not in_true_space('clusters', 'pandoraSCE::Reco2')


def test_displaced_copies_the_container():
    m = small_map()
    points = Points(id=[1, 2], x=[0.5, -0.5], y=[0.5, 0.5], z=[0.5, 0.5])
    moved = displaced(points, m)
    assert moved is not points and list(points.x) == [0.5, -0.5]
    np.testing.assert_allclose(moved.x, [0.5, 0.5])
    np.testing.assert_allclose(moved.z, [1.0, 0.5])
    assert list(moved.id) == [1, 2] and moved.x.dtype == points.x.dtype
    # showers move by their start
    showers = Showers.from_arrays({**Showers.empty().to_arrays(), **{
        c: [0.5] for c in ('start_x', 'start_y', 'start_z', 'dir_x', 'dir_y', 'dir_z', 'length', 'open_angle')},
        'id': [3], 'best_plane': [2]})
    moved = displaced(showers, m)
    assert moved.start_z[0] == pytest.approx(1.0) and moved.dir_z[0] == pytest.approx(0.5)
    # nothing to move: the same container
    assert displaced(Points.empty(), m) is not None
    hits = Hits.empty()
    assert displaced(hits, m) is hits


def test_the_map_is_pandoras_correction(event, sbnd_map):
    ''' pandoraSCE spacepoints moved by the forward map: where pandora (uncorrected) put them '''
    stored, corrected = (event._read(0, 'spacepoints', f'{t}::Reco2') for t in ('pandora', 'pandoraSCE'))
    row = {i: k for k, i in enumerate(corrected.id.tolist())}
    pairs = np.array([(k, row[i]) for k, i in enumerate(stored.id.tolist()) if i in row])
    moved = displaced(corrected, sbnd_map)
    xyz = lambda p, rows: np.c_[p.x[rows], p.y[rows], p.z[rows]].astype(float)  # noqa: E731
    correction = xyz(stored, pairs[:, 0]) - xyz(corrected, pairs[:, 1])
    residual = xyz(stored, pairs[:, 0]) - xyz(moved, pairs[:, 1])
    assert len(pairs) > 1000
    # a few mm, of corrections over a cm (mostly in y and z)
    assert np.all(np.sqrt((residual ** 2).mean(axis=0)) < 0.4)
    length = lambda d: np.sqrt((d ** 2).sum(axis=1).mean())  # noqa: E731
    assert length(residual) < 0.3 * length(correction) and length(correction) > 1


def test_mc_tracks_land_on_their_hits(event, sbnd_map):
    ''' the neutrino's MC tracks: within a wire and 3 ticks of a hit, far more often with the map '''
    geom = Geometry.load('sbnd', 'sbnd_v02_04')
    hits = event._read(0, 'hits', 'gaushit::Reco1')
    mctracks = event._read(0, 'mctracks', 'mcreco::G4')

    def on_hits(mct, plane):
        layout = ViewLayout(geom, plane)
        hp = hit_positions(layout, hits)
        tracks = mctracks_as_tracks(mct)
        points = []
        for row in np.flatnonzero(mct.origin == 1):
            correction = track_correction(layout, tracks, row, t0=mc_t0(geom, mct.start_t[row]))
            points += [np.c_[s.x, s.y] for s in track_segments(layout, tracks, row, correction)]
        points = np.vstack(points)
        return np.mean([np.any((np.abs(hp.x - x) <= 1) & (np.abs(hp.y - y) <= 3)) for x, y in points])

    for plane in range(3):
        before, after = on_hits(mctracks, plane), on_hits(displaced(mctracks, sbnd_map), plane)
        assert after > 0.75 and after - before > 0.15, (plane, before, after)
