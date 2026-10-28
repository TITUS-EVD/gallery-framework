"""
TPC view layout, wire images and hit positions (titus.display), checked
against the legacy concatenation in titus/drawables/wire.py and against each
other on real SBND data (tests/data/sbnd-reco1-trimmed, made with
tools/trim_dump.py).
"""

import pathlib

import numpy as np
import pytest

from titus.display import ViewLayout, hit_positions, image_n_ticks, plane_index_table, wire_image
from titus.geometry import Geometry
from titus.readers import DumpReader, Hits, WireSignals

DUMP = pathlib.Path(__file__).parent / 'data' / 'sbnd-reco1-trimmed'
ICARUS_DUMP = pathlib.Path(__file__).parent / 'data' / 'icarus-stage0-trimmed'


@pytest.fixture(scope='module', params=['sbnd', 'icarus'])
def geom(request):
    return Geometry.load(request.param)


@pytest.fixture(scope='module')
def sbnd():
    return Geometry.load('sbnd')


@pytest.fixture(scope='module')
def dump():
    return DumpReader(DUMP)


def views(geom):
    return [(p, c) for c in range(geom.nCryos()) for p in range(geom.nPlanes())]


# --- legacy reference ---

def legacy_plane_arrays(geom, signals, n_ticks):
    """ per TITUS plane index, the (wires, ticks) array the C++ DrawWire fills """
    cm = geom.detector.channel_map
    arrays = [np.zeros((p.n_wires, n_ticks), dtype='f4') for p in geom.detector.planes]
    for row in range(len(signals)):
        dense = signals.dense(row)[:n_ticks]
        for cryo, tpc, plane, wire in cm.wires(int(signals.channel[row])):
            arrays[geom.getPlaneID(plane, tpc, cryo)][wire, :len(dense)] = dense
    return arrays


def legacy_get_plane(geom, arrays, plane, cryo):
    """ titus/drawables/wire.py Wire.getPlane and _concatenatePlanes """
    gap = geom.cathodeGap()

    def concatenate(right_plane, left_plane):
        array_right, array_left = arrays[right_plane], arrays[left_plane]
        if geom.flipPlane(left_plane):
            array_left = np.flip(array_left, axis=1)
        array_right = np.pad(array_right, pad_width=((0, 0), (0, int(gap))), mode='constant')
        return np.concatenate((array_right, array_left), axis=1)

    plane = geom.getPlaneID(plane=plane, tpc=0, cryo=cryo)
    other_planes = geom.getOtherPlanes(plane_id=plane)
    if len(other_planes) == 0:
        return arrays[plane]
    if len(other_planes) == 1:
        return concatenate(plane, other_planes[0])
    array_1 = concatenate(plane, other_planes[1])
    array_2 = concatenate(other_planes[0], other_planes[2])
    array_1 = np.pad(array_1, pad_width=((0, int(gap)), (0, 0)), mode='constant')
    return np.concatenate((array_1, array_2), axis=0)


def random_signals(geom, n_channels=300, n_ticks=None, seed=1):
    rng = np.random.default_rng(seed)
    n_ticks = n_ticks or geom.tRange()
    channels = np.sort(rng.choice(geom.detector.channel_map.n_channels, n_channels, replace=False))
    rois = []
    for _ in channels:
        starts = np.sort(rng.choice(n_ticks - 50, 2, replace=False))
        rois.append([(int(s), rng.uniform(1, 100, rng.integers(1, 50))) for s in starts])
    return WireSignals.from_rois(channels, [n_ticks] * n_channels, rois)


# --- layout ---

def test_sbnd_layout(sbnd):
    layout = ViewLayout(sbnd, plane=0)
    gap = int(sbnd.cathodeGap())
    assert [(b.plane_index, b.wire_offset, b.time_offset, b.flip) for b in layout.blocks] == \
        [(0, 0, 0, False), (4, 0, 3400 + gap, True)]
    assert layout.shape == (sbnd.detector.planes[0].n_wires, 2 * 3400 + gap)


def test_icarus_layout():
    icarus = Geometry.load('icarus')
    gap = int(icarus.cathodeGap())
    for cryo, first in [(0, 0), (1, 12)]:
        layout = ViewLayout(icarus, plane=0, cryostat=cryo)
        n_wires = icarus.detector.planes[first].n_wires
        # TPCs 0/1 and 2/3 share a drift volume, split at z = 0
        assert [(b.plane_index - first, b.wire_offset, b.time_offset, b.flip) for b in layout.blocks] == [
            (0, 0, 0, False), (3, n_wires + gap, 0, False),
            (6, 0, 4096 + gap, True), (9, n_wires + gap, 4096 + gap, True)]
        assert layout.shape == (2 * n_wires + gap, 2 * 4096 + gap)


def test_layout_n_ticks_override(sbnd):
    layout = ViewLayout(sbnd, plane=2, n_ticks=3427)
    assert layout.blocks[1].time_offset == 3427 + int(sbnd.cathodeGap())
    assert layout.shape[1] == 2 * 3427 + int(sbnd.cathodeGap())


def test_tick_to_view(sbnd):
    layout = ViewLayout(sbnd, plane=0)
    assert layout.tick_to_view(0, 10.5) == 10.5
    # flipped: the start of the readout is the end of the second block
    end = layout.shape[1]
    assert layout.tick_to_view(4, 0) == end
    assert layout.tick_to_view(4, 3400) == layout.blocks[1].time_offset
    assert layout.wire_to_view(4, 7) == 7.5


def test_plane_index_table(geom):
    table = plane_index_table(geom)
    for index, p in enumerate(geom.detector.planes):
        assert table[p.id.cryostat, p.id.tpc, p.id.plane] == index


# --- wire images ---

@pytest.mark.parametrize('n_ticks', [None, 3000])
def test_wire_image_matches_legacy(geom, n_ticks):
    n = n_ticks or geom.tRange()
    signals = random_signals(geom, n_ticks=geom.tRange())
    arrays = legacy_plane_arrays(geom, signals, n)
    for plane, cryo in views(geom):
        image = wire_image(ViewLayout(geom, plane, cryo, n_ticks=n), signals, geom.detector.channel_map)
        np.testing.assert_array_equal(image, legacy_get_plane(geom, arrays, plane, cryo))


def test_wire_image_matches_legacy_on_data(sbnd, dump):
    signals = dump.wires(0, 'simtpc2d:dnnsp')
    n_ticks = int(signals.n_ticks.max())
    assert n_ticks == 3427
    arrays = legacy_plane_arrays(sbnd, signals, n_ticks)
    for plane, cryo in views(sbnd):
        image = wire_image(ViewLayout(sbnd, plane, cryo, n_ticks=n_ticks), signals, sbnd.detector.channel_map)
        assert image.any()
        np.testing.assert_array_equal(image, legacy_get_plane(sbnd, arrays, plane, cryo))


def test_icarus_channel_rois_match_legacy_on_data():
    """ one ChannelROI product per drift volume, added into the split-wire views """
    icarus = Geometry.load('icarus')
    reader = DumpReader(ICARUS_DUMP)
    products = [reader.channel_rois(0, tag) for tag in reader.tags('channel_rois', 0)]
    assert len(products) == 4
    n_ticks = image_n_ticks(icarus, *products)
    assert n_ticks == 4096
    per_product = [legacy_plane_arrays(icarus, signals, n_ticks) for signals in products]
    arrays = [sum(a[i] for a in per_product) for i in range(len(icarus.detector.planes))]
    filled = []
    for plane, cryo in views(icarus):
        layout = ViewLayout(icarus, plane, cryo, n_ticks=n_ticks)
        image = sum(wire_image(layout, signals, icarus.detector.channel_map) for signals in products)
        filled += [icarus.detector.planes[b.plane_index].id[:2] for b in layout.blocks
                   if image[layout.block_slices(b.plane_index)].any()]
        np.testing.assert_array_equal(image, legacy_get_plane(icarus, arrays, plane, cryo))
    # every TPC of both cryostats contributes signal to some view
    assert set(filled) == {(c, t) for c in range(2) for t in range(4)}


def test_image_n_ticks(sbnd):
    full = WireSignals.from_rois([1], [3427], [[(0, [1.])]])
    short = WireSignals.from_rois([1, 2], [10, 0], [[(5, [1.] * 5)], []])
    assert image_n_ticks(sbnd) == 3400
    assert image_n_ticks(sbnd, full) == 3427
    # a ChannelROI waveform ends at its last ROI: never shrink the window
    assert image_n_ticks(sbnd, short) == 3400
    assert image_n_ticks(sbnd, WireSignals.empty()) == 3400


def test_wire_image_drops_samples_past_window(sbnd):
    """ the C++ DrawWire copies them into the next wire's row instead """
    cm = sbnd.detector.channel_map
    channel = int(cm.channel[(cm.tpc == 0) & (cm.plane == 2) & (cm.wire == 100)][0])
    signals = WireSignals.from_rois([channel], [3427], [[(3398, [1., 2., 3., 4., 5.])]])
    image = wire_image(ViewLayout(sbnd, plane=2, n_ticks=3400), signals, cm)
    assert image[100, 3398:3400].tolist() == [1, 2]
    assert image.sum() == 3
    assert not image[101].any()


def test_wire_image_split_wire_channels():
    """ ICARUS induction-2/collection channels read out wires in two TPCs """
    icarus = Geometry.load('icarus')
    cm = icarus.detector.channel_map
    counts = np.bincount(cm.channel)
    channel = int(np.flatnonzero(counts == 2)[0])
    (c, t0, p, w0), (_, t1, _, w1) = cm.wires(channel)
    signals = WireSignals.from_rois([channel], [4096], [[(1000, [5.])]])
    layout = ViewLayout(icarus, plane=p, cryostat=c)
    image = wire_image(layout, signals, cm)
    filled = sorted(zip(*np.nonzero(image)))
    expected = sorted((int(layout.wire_to_view(icarus.getPlaneID(p, t, c), w)),
                       int(layout.tick_to_view(icarus.getPlaneID(p, t, c), 1000.5)))  # center of sample 1000
                      for t, w in [(t0, w0), (t1, w1)])
    assert filled == expected


def test_empty_signals(sbnd):
    image = wire_image(ViewLayout(sbnd, plane=0), WireSignals.empty(), sbnd.detector.channel_map)
    assert image.shape == ViewLayout(sbnd, plane=0).shape and not image.any()


# --- hits ---

def test_hits_peak_on_their_waveform(sbnd, dump):
    """
    Each gaushit peak is at the maximum of its waveform in the view image,
    for both TPCs: within one tick, and with no systematic offset (sample t
    covers [t, t + 1) in hit times). A one-tick shift fails this.
    """
    signals = dump.wires(0, 'simtpc2d:dnnsp')
    hits = dump.hits(0, 'gaushit')
    n_ticks = int(signals.n_ticks.max())
    shown = 0
    for plane, cryo in views(sbnd):
        layout = ViewLayout(sbnd, plane, cryo, n_ticks=n_ticks)
        image = wire_image(layout, signals, sbnd.detector.channel_map)
        positions = hit_positions(layout, hits)
        shown += len(positions)
        tpcs = hits.tpc[positions.index]
        for tpc in np.unique(tpcs):
            offsets = []
            for x, y, rms in zip(positions.x[tpcs == tpc], positions.y[tpcs == tpc],
                                 positions.half_height[tpcs == tpc]):
                lo = max(int(y - 2 * rms), 0)
                offsets.append(lo + np.argmax(image[int(x), lo:int(y + 2 * rms) + 1]) + 0.5 - y)
            offsets = np.array(offsets)
            assert (np.abs(offsets) < 1).mean() > 0.9, (plane, tpc)
            assert abs(np.median(offsets)) < 0.2, (plane, tpc)
    assert shown == len(hits)


def test_hit_positions(sbnd):
    hits = Hits(channel=[0, 0, 0], cryostat=[0, 0, 0], tpc=[0, 1, 1], plane=[0, 1, 0],
                wire=[10, 20, 30], peak_time=[100., 200., 300.], rms=[2., 3., 4.],
                integral=[1, 1, 1], peak_amplitude=[1, 1, 1], start_tick=[0, 0, 0], end_tick=[0, 0, 0])
    layout = ViewLayout(sbnd, plane=0)
    pos = hit_positions(layout, hits)
    # view U shows U of TPC 0 and V of TPC 1, whose wires are parallel to it
    assert pos.index.tolist() == [0, 1]
    assert pos.x.tolist() == [10.5, 20.5]
    assert pos.y.tolist() == [100., layout.tick_to_view(4, 200.)]
    assert pos.half_height.tolist() == [2., 3.]
    assert hit_positions(layout, hits, time_offset_ticks=5).y[0] == 105.
    assert len(hit_positions(layout, Hits.empty())) == 0


# --- unite cathodes ---

def unite(image, layout):
    """ the image as WireView shows it with "unite cathodes" """
    n, n_ticks, gap = layout.removed_ticks, layout.n_ticks, layout.gap
    return np.delete(image, np.r_[n_ticks - n:n_ticks, n_ticks + gap:n_ticks + gap + n], axis=1)


def test_united_layout(sbnd):
    layout = ViewLayout(sbnd, plane=0, removed_ticks=100)
    full = ViewLayout(sbnd, plane=0)
    n_ticks = layout.n_ticks
    # first TPC: unchanged before the cut, hidden in it
    assert layout.tick_to_view(0, 1000.) == full.tick_to_view(0, 1000.)
    assert np.isnan(layout.tick_to_view(0, n_ticks - 50.))
    assert layout.tick_to_view(0, n_ticks - 50., hide_removed=False) == n_ticks - 100
    # second TPC (flipped): its late ticks are cut, the rest moves up by both cuts
    assert np.isnan(layout.tick_to_view(4, n_ticks - 50.))
    assert layout.tick_to_view(4, 1000.) == full.tick_to_view(4, 1000.) - 200
    # the cathodes meet across the gap
    cathode = n_ticks - 100.
    gap = layout.tick_to_view(4, cathode) - layout.tick_to_view(0, cathode)
    assert gap == pytest.approx(layout.gap)


def test_united_hits_peak_on_their_waveform(sbnd, dump):
    """ the cut image (as WireView shows it) and the united layout agree """
    signals = dump.wires(0, 'simtpc2d:dnnsp')
    hits = dump.hits(0, 'gaushit')
    for plane in range(3):
        cathode = sbnd.anodeCathodeTicks(plane)[0][1]
        layout = ViewLayout(sbnd, plane, removed_ticks=int(sbnd.tRange() - cathode))
        image = unite(wire_image(ViewLayout(sbnd, plane), signals, sbnd.detector.channel_map), layout)
        positions = hit_positions(layout, hits)
        assert len(positions) > 0
        offsets = []
        for x, y, rms in zip(positions.x, positions.y, positions.half_height):
            lo = max(int(y - 2 * rms), 0)
            offsets.append(lo + np.argmax(image[int(x), lo:int(y + 2 * rms) + 1]) + 0.5 - y)
        offsets = np.array(offsets)
        assert (np.abs(offsets) < 1).mean() > 0.9
        assert abs(np.median(offsets)) < 0.2


def test_united_hides_hits_past_the_cathode(sbnd):
    hits = Hits(channel=[0, 0], cryostat=[0, 0], tpc=[0, 0], plane=[0, 0], wire=[10, 20],
                peak_time=[1000., 3300.], rms=[1., 1.], integral=[1, 1], peak_amplitude=[1, 1],
                start_tick=[0, 0], end_tick=[0, 0])
    positions = hit_positions(ViewLayout(sbnd, plane=0, removed_ticks=300), hits)
    assert positions.index.tolist() == [0]
