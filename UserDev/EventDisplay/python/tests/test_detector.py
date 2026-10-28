"""
The lightweight detector description must reproduce what the LArSoft services
computed (golden tables from tools/dump_detector.py).
"""

import subprocess
import sys

import numpy as np
import pytest

from titus.geometry import PlaneID, load_detector


def projections(golden):
    for rec in golden['projections']:
        for pr in rec['planes']:
            yield np.array(rec['point']), pr


def test_import_does_not_need_root():
    code = ("import sys, titus.geometry; "
            "assert 'ROOT' not in sys.modules and 'PyQt5' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], check=True)


def test_wire_coordinate_matches_larsoft(detector_and_golden):
    det, golden = detector_and_golden
    for point, pr in projections(golden):
        expected = pr['wire_coordinate']
        assert det.plane(*pr['id']).wire_coordinate(point) == pytest.approx(expected, abs=1e-6)


def test_nearest_wire_matches_larsoft(detector_and_golden):
    det, golden = detector_and_golden
    for point, pr in projections(golden):
        expected = pr['nearest_wire'] if pr['nearest_wire'] is not None else -1
        assert det.plane(*pr['id']).nearest_wire(point) == expected, (point, pr)


def test_ticks_match_larsoft(detector_and_golden):
    det, golden = detector_and_golden
    for point, pr in projections(golden):
        assert det.plane(*pr['id']).x_to_ticks(point[0]) == pytest.approx(pr['ticks'], abs=1e-6)


def test_ticks_round_trip(detector_and_golden):
    det, _ = detector_and_golden
    ticks = np.linspace(0, det.timing.number_time_samples, 7)
    for plane in det.planes:
        np.testing.assert_allclose(plane.x_to_ticks(plane.ticks_to_x(ticks)), ticks)


def test_trigger_offset_tick_is_at_collection_plane(detector_and_golden):
    """ LArSoft's convention: tick == trigger offset at the last (collection) plane """
    det, _ = detector_and_golden
    for tpc in det.tpcs:
        collection = tpc.planes[-1]
        assert collection.x_to_ticks(collection.center[0]) == \
            pytest.approx(det.timing.trigger_offset_tpc_ticks, abs=1e-6)


def test_tpc_lookup_matches_larsoft(detector_and_golden):
    det, golden = detector_and_golden
    recs = golden['tpc_lookup'] + golden['projections']
    found = det.tpc_of(np.array([r['point'] for r in recs]))
    for rec, tpc in zip(recs, found):
        expected = rec['tpc'] if rec['tpc'] is not None else [-1, -1]
        assert list(tpc) == expected


def test_tpc_lookup_single_point(sbnd):
    assert tuple(sbnd.tpc_of([-100., 0., 250.])) == (0, 0)
    assert tuple(sbnd.tpc_of([100., 0., 250.])) == (0, 1)
    assert tuple(sbnd.tpc_of([1000., 0., 250.])) == (-1, -1)


def test_parallel_planes_sbnd(sbnd):
    """ U in TPC 0 has the wire angle of V in TPC 1, and vice versa """
    assert sbnd.parallel_planes((0, 0, 0)) == [PlaneID(0, 1, 1)]
    assert sbnd.parallel_planes((0, 0, 1)) == [PlaneID(0, 1, 0)]
    assert sbnd.parallel_planes((0, 0, 2)) == [PlaneID(0, 1, 2)]


def test_parallel_planes_icarus(icarus):
    for cryo in (0, 1):
        assert icarus.parallel_planes((cryo, 0, 0)) == [PlaneID(cryo, 1, 0), PlaneID(cryo, 2, 0), PlaneID(cryo, 3, 0)]
        assert icarus.parallel_planes((cryo, 0, 1)) == [PlaneID(cryo, 1, 1), PlaneID(cryo, 2, 2), PlaneID(cryo, 3, 2)]
        assert icarus.parallel_planes((cryo, 0, 2)) == [PlaneID(cryo, 1, 2), PlaneID(cryo, 2, 1), PlaneID(cryo, 3, 1)]


def test_channel_map_sbnd(sbnd):
    cm = sbnd.channel_map
    assert cm.n_channels == 11276
    # every channel reads out exactly one wire
    assert len(cm.channel) == cm.n_channels
    assert cm.wires(0) == [(0, 0, 0, 0)]


def test_channel_map_icarus_split_wires(icarus):
    """ Induction-2 and collection channels near z = 0 read wires in two TPCs """
    cm = icarus.channel_map
    channels, counts = np.unique(cm.channel, return_counts=True)
    multi = set(channels[counts > 1])
    assert len(multi) == 4224
    for ch in list(multi)[:50]:
        wires = cm.wires(ch)
        assert len(wires) == 2
        (c0, t0, p0, _), (c1, t1, p1, _) = wires
        assert c0 == c1 and t0 != t1 and p0 == p1 != 0


def test_opdets(sbnd, icarus):
    assert len(sbnd.opdets) == 312
    assert set(sbnd.opdets.shape) == {'TGeoBBox', 'TGeoSphere'}
    assert len(icarus.opdets) == 360
    assert set(icarus.opdets.shape) == {'TGeoSphere'}


def test_unknown_detector():
    with pytest.raises(LookupError):
        load_detector('microboone')
    with pytest.raises(LookupError):
        load_detector('sbnd', gdml='sbnd_v99')
