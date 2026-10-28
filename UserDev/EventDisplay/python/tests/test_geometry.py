"""
Regression tests: the Geometry built from detector JSON + display settings
must reproduce the hand-coded values of the legacy sbnd/icarus classes in
titus/gallery_interface/geometry.py (ICARUS in its default split-wire mode).
Deliberate differences are marked with LEGACY_DIFFERS.
"""

import math

import numpy as np
import pytest

from titus.geometry import Geometry, ServiceHandles

# values copied from titus/gallery_interface/geometry.py
LEGACY = {
    'sbnd': {
        'planeid_map_size': 6,
        'plane_flip': [False, False, False, True, True, True],
        'plane_shift': [False] * 6,
        'other_planes': {0: [4], 1: [3], 2: [5]},
        'plane_mix': {0: [3], 1: [4], 2: [5]},
        'projections_match': False,
        'view_names': ['U', 'V', 'Y'],
        'levels': [(100, -10), (-10, 100), (-10, 200)],
        'pedestals': [0, 0, 0],
        'cathode_gap_cm': 5.3,
        'n_tpcs': 2,
        'n_cryos': 1,
        'n_planes': 3,
        'n_views': 6,
        'opdet_radius': 6,
        'n_optical_frames': 3,
        'logo': 'SBND-color.png',
        'logo_scale': 0.13,
        'crt': (400.0, -376.0, 746.0, -177.5),
        'apply_trigger_offset': True,
    },
    'icarus': {
        'planeid_map_size': 24,
        'plane_flip': [False] * 6 + [True] * 6 + [False] * 6 + [True] * 6,
        'plane_shift': ([False] * 3 + [True] * 3) * 4,
        'other_planes': {0: [3, 6, 9], 1: [4, 8, 11], 2: [5, 7, 10],
                         12: [15, 18, 21], 13: [16, 20, 23], 14: [17, 19, 22]},
        'plane_mix': {0: [6], 1: [8], 2: [7], 12: [18], 13: [20], 14: [19]},
        'projections_match': False,
        'view_names': ['H', 'U', 'V'],
        'levels': [(100, 0), (0, 100), (0, 100), (100, 0), (0, 100), (0, 100)],
        'pedestals': [0] * 6,
        'cathode_gap_cm': 6.0,
        'n_tpcs': 2,
        'n_cryos': 2,
        'n_planes': 3,
        'n_views': 24,
        'opdet_radius': 7,
        'n_optical_frames': 1,
        'logo': 'logo_icarus.png',
        'logo_scale': 0.35,
        'crt': (None, None, None, None),
        'apply_trigger_offset': False,
    },
}

# the parallel-wire rule pairs SBND's U in TPC 0 with V in TPC 1, as the
# legacy getOtherPlanes (used by hits and wires) already did; the legacy
# planeMix (used by showers) paired planes by number instead
LEGACY_DIFFERS = {('sbnd', 'plane_mix'): {0: [4], 1: [3], 2: [5]}}


def legacy_opdet_to_tpc(name, x):
    """ legacy sbnd/icarus opdetToTPC() """
    if name == 'sbnd':
        return 0 if x < 0 else 1
    if x < -100:
        return 0
    if -100 < x < 0:
        return 1
    if 0 < x < 100:
        return 2
    return 3


@pytest.fixture(scope='module', params=['sbnd', 'icarus'])
def geom(request):
    return Geometry.load(request.param)


def legacy(geom, key):
    return LEGACY_DIFFERS.get((geom.name(), key), LEGACY[geom.name()][key])


def test_plane_numbering(geom):
    assert len(geom._planeid_map) == legacy(geom, 'planeid_map_size')
    assert geom.getPlaneID(plane=0, tpc=0, cryo=0) == 0
    assert geom.getPlaneID(plane=2, tpc=1, cryo=0) == 5
    with pytest.raises(LookupError):
        geom.getPlaneID(plane=0, tpc=9, cryo=0)


@pytest.mark.parametrize('key, method', [
    ('plane_flip', 'planeFlip'),
    ('plane_mix', 'planeMix'),
    ('projections_match', 'projectionsMatch'),
    ('view_names', 'viewNames'),
    ('n_tpcs', 'nTPCs'),
    ('n_cryos', 'nCryos'),
    ('n_planes', 'nPlanes'),
    ('n_views', 'nViews'),
    ('opdet_radius', 'opdetRadius'),
    ('n_optical_frames', 'nOpticalFrames'),
    ('logo_scale', 'logoScale'),
])
def test_matches_legacy(geom, key, method):
    assert getattr(geom, method)() == legacy(geom, key)


def test_plane_shift(geom):
    assert [geom.shiftPlane(i) for i in range(geom.nViews())] == legacy(geom, 'plane_shift')


def test_other_planes(geom):
    for plane, expected in legacy(geom, 'other_planes').items():
        assert geom.getOtherPlanes(plane) == expected


def test_levels_and_pedestals(geom):
    levels = legacy(geom, 'levels')
    assert [geom.getLevels(p) for p in range(len(levels))] == levels
    assert geom._pedestals == legacy(geom, 'pedestals')


def test_timing_and_gap(geom):
    timing = geom.detector.timing
    # legacy: TickPeriod * DriftVelocity, and the trigger offset in TPC ticks
    assert geom.time2cm() == pytest.approx(timing.tpc_tick_period * timing.drift_velocity)
    assert geom.cathodeGap() == pytest.approx(legacy(geom, 'cathode_gap_cm') / geom.time2cm())
    assert geom.samplingRate() == pytest.approx(timing.tpc_tick_period * 1000)
    assert geom.aspectRatio() == pytest.approx(geom.wire2cm() / geom.time2cm())
    expected_offset = geom.triggerOffset() * geom.time2cm() if legacy(geom, 'apply_trigger_offset') else 0
    for plane in range(geom.nViews()):
        assert geom.offset(plane) == pytest.approx(expected_offset)


def test_specific_values():
    sbnd, icarus = Geometry.load('sbnd'), Geometry.load('icarus')
    assert (sbnd.triggerOffset(), sbnd.tRange(), sbnd.wire2cm()) == (410, 3400, pytest.approx(0.3))
    assert (icarus.triggerOffset(), icarus.tRange(), icarus.wire2cm()) == (850, 4096, pytest.approx(0.3))
    assert sbnd.time2cm() == pytest.approx(0.0781334, rel=1e-5)
    assert icarus.time2cm() == pytest.approx(0.0628576, rel=1e-5)


def test_wire_range(geom):
    """ legacy: wires in the planes of TPC 0; doubled plus the gap with split wires """
    tpc0 = geom.detector.tpc(0, 0)
    for i, plane in enumerate(tpc0.planes):
        expected = plane.n_wires * 2 + geom.cathodeGap() if geom.splitWire() else plane.n_wires
        assert geom.wRange(i) == pytest.approx(expected)


def test_opdets(geom):
    x, y, z = geom.opdetLoc()
    n = len(geom.detector.opdets)
    assert len(x) == len(y) == len(z) == len(geom.opdetName()) == n
    for ch in range(n):
        assert geom.opdetToTPC(ch) == legacy_opdet_to_tpc(geom.name(), x[ch])
    assert math.isnan(geom.opdetDefaultValue())


def test_opdet_names():
    sbnd, icarus = Geometry.load('sbnd'), Geometry.load('icarus')
    assert set(sbnd.opdetName()) == {'pmt_coated', 'pmt_uncoated', 'xarapuca_vis', 'xarapuca_vuv'}
    # legacy ICARUS names came from the op det shape
    assert set(icarus.opdetName()) == {'pmt_coated'}


def test_crt(geom):
    crt = (geom.crt_top_ymin, geom.crt_bot_ymax, geom.crt_front_zmin, geom.crt_back_zmax)
    assert crt == legacy(geom, 'crt')


def test_color_maps(geom):
    n_default = 6
    for plane in range(n_default):
        state = geom.colorMap(plane)
        assert state['mode'] == 'rgb'
        assert all(isinstance(t, tuple) and isinstance(t[1], tuple) for t in state['ticks'])
    assert geom.colorMap(0, 'grayscale')['ticks']


def test_logo(geom):
    assert geom.logo().endswith(legacy(geom, 'logo'))
    assert not geom.hasLogo()


def test_services_missing_is_explicit(geom):
    with pytest.raises(RuntimeError, match='not loaded'):
        geom.getGeometryCore()


def test_override_time_range(geom):
    g = Geometry(geom.detector, geom.display)
    g.override_time_range(1234)
    assert g.tRange() == g.readoutWindowSize() == 1234


def test_anode_cathode_ticks(geom):
    """ anode at the trigger offset, cathode one full drift later, flips as the views """
    for cryo in range(geom.nCryos()):
        for plane in range(geom.nPlanes()):
            ticks = geom.anodeCathodeTicks(plane, cryo)
            assert len(ticks) == geom.nTPCs()
            assert [flipped for _, _, flipped in ticks] == [False, True]
            for anode, cathode, _ in ticks:
                assert anode == pytest.approx(geom.triggerOffset(), abs=0.5)
                assert 0 < anode < cathode < geom.tRange()


def test_anode_cathode_drift_length():
    sbnd, icarus = Geometry.load('sbnd'), Geometry.load('icarus')
    # SBND: collection plane at |x| = 202.05 cm, cathode at x = 0
    anode, cathode, _ = sbnd.anodeCathodeTicks(0)[0]
    assert (cathode - anode) * sbnd.time2cm() == pytest.approx(202.05, abs=0.01)
    # ICARUS: 149 cm drift
    anode, cathode, _ = icarus.anodeCathodeTicks(0)[0]
    assert (cathode - anode) * icarus.time2cm() == pytest.approx(149.04, abs=0.01)


def test_services_load_on_first_use(geom):
    calls = []

    def loader():
        calls.append(1)
        return ServiceHandles(geometry_core='core', detector_clocks='clocks')

    g = Geometry(geom.detector, geom.display, service_loader=loader)
    assert calls == []
    assert g.getGeometryCore() == 'core' and g.getDetectorClocks() == 'clocks'
    assert calls == [1]
    # loaded once; a service the loader didn't provide is still an error
    with pytest.raises(RuntimeError, match='not loaded'):
        g.getWireReadout()
    assert calls == [1]
