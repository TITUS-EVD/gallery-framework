"""
Wire and hit drawables on the reader path, offscreen, with products from the
test dumps in place of a gallery event. The other drawables still import the
C++ evd library, so ROOT (and galleryUtils, for titus.modules) is stubbed.
"""

import pathlib
import sys
from unittest import mock

import numpy as np
import pytest

sys.modules.setdefault('ROOT', mock.MagicMock())
sys.modules.setdefault('galleryUtils', mock.MagicMock())

pg = pytest.importorskip('pyqtgraph')
from PyQt5 import QtCore  # noqa: E402

import titus.drawables.drawable as drawable_module  # noqa: E402
from titus.display import ViewLayout, track_correction, track_segments, track_t0s, wire_image  # noqa: E402
from titus.drawables import Hit, RawDigit, RecoWire, Track, recoChannelROI  # noqa: E402
from titus.geometry import Geometry  # noqa: E402
from titus.readers import DumpReader, ProductNotFound, RawDigits  # noqa: E402

DATA = pathlib.Path(__file__).parent / 'data'


class FakeGalleryInterface(QtCore.QObject):
    eventChanged = QtCore.pyqtSignal()
    fileChanged = QtCore.pyqtSignal()

    def event_handle(self):
        return 'event'


@pytest.fixture
def products(monkeypatch):
    """ {(kind, tag): container} served by read_event_product """
    served = {}

    def read(lib, ev, kind, tag, columns=None, cpp_type=None, candidate_tags=None):
        if (kind, tag) not in served:
            raise ProductNotFound(f'{kind} {tag} not in this event')
        return served[(kind, tag)]

    monkeypatch.setattr(drawable_module, 'read_event_product', read)
    monkeypatch.setattr(drawable_module, 'titusio', lambda: None)
    # the file's products: whatever is served
    monkeypatch.setattr(drawable_module, 'file_products', lambda ev: {key: 'type' for key in served})
    monkeypatch.setattr(drawable_module, 'string_vector', list)
    return served


@pytest.fixture(scope='module')
def sbnd():
    return Geometry.load('sbnd')


def test_reco_wire(qtbot, products, sbnd):
    dump = DumpReader(DATA / 'sbnd-reco1-trimmed')
    signals = dump.wires(0, 'simtpc2d:dnnsp')
    products[('wires', 'simtpc2d:dnnsp:DetSim')] = signals
    drawer = RecoWire(FakeGalleryInterface(), sbnd)
    drawer.set_producer('simtpc2d:dnnsp:DetSim')
    drawer.analyze()
    for plane in range(3):
        image = drawer.getPlane(plane, 0)
        layout = ViewLayout(sbnd, plane)
        # the views assume geom.tRange() ticks per TPC
        assert image.shape == layout.shape == (sbnd.detector.planes[plane].n_wires, 2 * 3400 + 67)
        np.testing.assert_array_equal(image, wire_image(layout, signals, sbnd.detector.channel_map))
        assert image.any()


def test_channel_rois_from_several_producers(qtbot, products):
    icarus = Geometry.load('icarus')
    dump = DumpReader(DATA / 'icarus-stage0-trimmed')
    tags = dump.tags('channel_rois', 0)
    for tag in tags:
        products[('channel_rois', tag)] = dump.channel_rois(0, tag)
    drawer = recoChannelROI(FakeGalleryInterface(), icarus)
    drawer.set_producer(tags)
    drawer.analyze()
    assert len(drawer._data) == 4
    for cryo in range(2):
        layout = ViewLayout(icarus, plane=0, cryostat=cryo)
        expected = sum(wire_image(layout, dump.channel_rois(0, t), icarus.detector.channel_map) for t in tags)
        np.testing.assert_array_equal(drawer.getPlane(0, cryo), expected)


def test_missing_product_is_skipped(qtbot, products, sbnd, capsys):
    drawer = RecoWire(FakeGalleryInterface(), sbnd)
    drawer.set_producer('caldata::Reco1')
    drawer.analyze()
    assert 'not in this event' in capsys.readouterr().out
    assert not drawer.getPlane(0, 0).any()


def test_no_producer_draws_nothing(qtbot, products, sbnd):
    drawer = RecoWire(FakeGalleryInterface(), sbnd)
    drawer.set_producer(None)
    drawer.analyze()
    assert drawer._data == [] and not drawer.getPlane(0, 0).any()


@pytest.mark.parametrize('subtract', [True, False])
def test_raw_digits(qtbot, products, sbnd, subtract):
    cm = sbnd.detector.channel_map
    channel = int(cm.channel[(cm.tpc == 0) & (cm.plane == 2) & (cm.wire == 10)][0])
    adc = np.full((1, 3400), 405, dtype='i2')
    adc[0, 100] = 500
    products[('raw_digits', 'daq::DetSim')] = RawDigits(channel=[channel], pedestal=[400.], n_samples=[3400], adc=adc)
    drawer = RawDigit(FakeGalleryInterface(), sbnd)
    drawer.setSubtractPedestal(subtract)
    drawer.set_producer('daq::DetSim')
    drawer.analyze()
    row = drawer.getPlane(2, 0)[10]
    pedestal = 400 if subtract else 0
    assert row[100] == 500 - pedestal and row[0] == 405 - pedestal
    assert not row[3400:].any()


class FakeView:
    def __init__(self, plane, cryostat, widget):
        self._plane, self._cryostat = plane, cryostat
        self._view = widget.addViewBox()

    def plane(self):
        return self._plane

    def cryostat(self):
        return self._cryostat


@pytest.fixture
def tpc_module(qtbot, sbnd):
    widget = pg.GraphicsLayoutWidget()
    qtbot.addWidget(widget)
    return mock.Mock(widget=widget, _wire_views={(p, 0): FakeView(p, 0, widget) for p in range(3)})


def test_hits(qtbot, products, sbnd, tpc_module):
    hits = DumpReader(DATA / 'sbnd-reco1-trimmed').hits(0, 'gaushit')
    products[('hits', 'gaushit::Reco1')] = hits
    drawer = Hit(FakeGalleryInterface(), sbnd, tpc_module)
    drawer.set_producer('gaushit::Reco1')
    drawer.analyze()
    drawer.drawObjects()

    boxes = [r for c in drawer._drawnObjects for r in c._rects]
    assert len(boxes) == len(hits)
    # view U, TPC 0: box centered on the wire and on the peak, +- RMS
    view = tpc_module._wire_views[(0, 0)]
    (collection,) = [c for c in drawer._drawnObjects if c in view._view.addedItems]
    first = hits.on_plane(0, 0, 0)
    rects = sorted((r.rect().x(), r.rect().center().y(), r.rect().height()) for r in collection._rects
                   if r.rect().center().y() < 3400)
    expected = sorted(zip(hits.wire[first] + 0.0, hits.peak_time[first].astype(float), 2. * hits.rms[first]))
    np.testing.assert_allclose(rects, expected, rtol=1e-5)

    drawer.clearDrawnObjects()
    assert drawer._drawnObjects == []
    assert not [i for i in view._view.addedItems if i in boxes]


def test_crt_view_from_detector_description(qtbot, sbnd):
    """ the CRT view is built from the dumped AuxDets, without services """
    pytest.importorskip('PIL')
    from titus.modules.crt_module import CrtViewWidget

    widget = CrtViewWidget(sbnd)
    qtbot.addWidget(widget)
    auxdets = sbnd.detector.auxdets
    assert len(widget._drawn_crt_modules) == len(auxdets)
    assert len(widget._crt_strips) == sum(len(a['strips']) for a in auxdets)
    # sipms 2i and 2i + 1 read strip i
    mac = auxdets[0]['local_mac']
    assert widget._crt_strip(mac, 6) == widget._crt_strip(mac, 7) == (mac, 3)
    low, high = widget._crt_strip_world_bounds((mac, 3))
    assert low.tolist() == auxdets[0]['strips'][3]['min'] and high.tolist() == auxdets[0]['strips'][3]['max']
    assert widget._crt_strip(mac, 999) is None


def test_services_load_only_when_asked(qtbot, monkeypatch, sbnd):
    pytest.importorskip('PIL')
    from titus.modules.larsoft_module import LArSoftModule

    created = []

    def service_manager(name):
        created.append(name)
        return mock.MagicMock(name=name)

    monkeypatch.setitem(sys.modules, 'SBNDservices', mock.Mock(ServiceManager=service_manager))
    lsm = LArSoftModule('sbnd')
    loaded = []
    lsm.servicesLoaded.connect(lambda: loaded.append(1))
    geom = Geometry(sbnd.detector, sbnd.display, service_loader=lsm.load_services)
    assert created == [] and not lsm.services_loaded

    geom.getWireReadout()
    assert lsm.services_loaded and loaded == [1]
    assert sorted(created) == sorted(['DetectorClocks', 'DetectorProperties', 'Geometry', 'WireReadout',
                                      'LArProperties', 'AuxDetGeometry'])
    assert geom.getGeometryCore() is lsm.geom_service
    geom.getDetectorClocks()
    assert len(created) == 6 and loaded == [1]


def test_lazy_drawable_created_on_first_producer():
    from titus.drawables import LazyDrawable

    created = []

    def factory():
        drawable = mock.Mock()
        created.append(drawable)
        return drawable

    lazy = LazyDrawable(factory, 'recob::Track')
    lazy.parent_module = 'module'
    # the module analyzes, draws and clears all its drawables every event
    lazy.analyze()
    lazy.drawObjects()
    lazy.clearDrawnObjects()
    lazy.set_producer(None)
    assert created == [] and lazy.product_name == 'recob::Track'
    with pytest.raises(AttributeError, match='not created yet'):
        lazy.getHitsOnWire

    lazy.set_producer('pandoraTrack::Reco2')
    (drawable,) = created
    assert drawable.parent_module == 'module'
    drawable.set_producer.assert_called_with('pandoraTrack::Reco2')
    lazy.analyze()
    lazy.drawObjects()
    drawable.analyze.assert_called_once()
    drawable.drawObjects.assert_called_once()
    lazy.set_producer(None)
    drawable.set_producer.assert_called_with(None)
    lazy.set_producer('pandoraTrack::Reco2')
    assert len(created) == 1



def serve_reco2(products, entry, with_associations=True):
    dump = DumpReader(DATA / 'sbnd-reco2-tracks')
    products[('tracks', 'pandoraTrack::Reco2')] = dump.get(entry, 'tracks', 'pandoraTrack')
    products[('hits', 'gaushit::Reco1')] = dump.hits(entry, 'gaushit')
    if with_associations:
        for kind, tag in [('track_hits', 'pandoraTrack::Reco2'), ('pfp_tracks', 'pandoraTrack::Reco2'),
                          ('pfp_t0s', 'pandora::Reco2'), ('t0s', 'pandora::Reco2')]:
            products[(kind, tag)] = dump.get(entry, kind, tag)
    return dump


def draw_tracks(tpc_module, sbnd, corrected=True):
    tpc_module.correct_3d_objects = lambda: corrected
    drawer = Track(FakeGalleryInterface(), sbnd, tpc_module)
    drawer.set_producer('pandoraTrack::Reco2')
    drawer.analyze()
    drawer.drawObjects()
    return drawer


def test_tracks_corrected(qtbot, products, sbnd, tpc_module, capsys):
    dump = serve_reco2(products, 1)
    drawer = draw_tracks(tpc_module, sbnd)
    # every related product was read
    assert 'Warning' not in capsys.readouterr().out
    tracks = dump.get(1, 'tracks', 'pandoraTrack')
    t0s = track_t0s(len(tracks), dump.get(1, 'pfp_tracks', 'pandoraTrack'),
                    dump.get(1, 'pfp_t0s', 'pandora'), dump.get(1, 't0s', 'pandora'))
    lines = drawer._drawnObjects
    dashed = [l for l in lines if l.pen().style() == QtCore.Qt.DashLine]
    assert lines and dashed
    assert any('T0 ' in l.toolTip() for l in dashed)
    # the collection view: same paths as the display code computes
    layout = ViewLayout(sbnd, plane=2)
    expected = [s for row in range(len(tracks))
                for s in track_segments(layout, tracks, row, track_correction(
                    layout, tracks, row, t0s[row], dump.get(1, 'track_hits', 'pandoraTrack'),
                    dump.hits(1, 'gaushit')))]
    view_lines = [l for l in lines if l in tpc_module._wire_views[(2, 0)]._view.addedItems]
    assert len(view_lines) == len(expected)
    first = view_lines[0].path().elementAt(0)
    assert (first.x, first.y) == pytest.approx((expected[0].x[0], expected[0].y[0]))

    drawer.clearDrawnObjects()
    assert drawer._drawnObjects == []


def test_tracks_as_stored(qtbot, products, sbnd, tpc_module):
    serve_reco2(products, 1)
    drawer = draw_tracks(tpc_module, sbnd, corrected=False)
    lines = drawer._drawnObjects
    assert lines and all(l.pen().style() == QtCore.Qt.SolidLine for l in lines)
    assert all('drawn as stored' in l.toolTip() for l in lines)


def test_tracks_without_associations(qtbot, products, sbnd, tpc_module, capsys):
    """ no T0s or track-hit associations in the file: no shift, TPC by readout window """
    serve_reco2(products, 1, with_associations=False)
    drawer = draw_tracks(tpc_module, sbnd)
    assert drawer._drawnObjects
    assert not any('T0 ' in l.toolTip() for l in drawer._drawnObjects)
    assert 'Warning' not in capsys.readouterr().out


@pytest.mark.parametrize('scale', [(1, 1), (20, 20), (20, 0.5), (0.05, 3)])
def test_line_hover_area_is_fixed_on_screen(qtbot, scale):
    """ zoomed in or out, and with unequal x and y scales: hover within HOVER_PIXELS of the line """
    from PyQt5 import QtGui, QtWidgets
    from titus.drawables.items import HOVER_PIXELS, polyLine

    scene = QtWidgets.QGraphicsScene()
    view = QtWidgets.QGraphicsView(scene)
    qtbot.addWidget(view)
    view.setTransform(QtGui.QTransform.fromScale(*scale))
    def at_pixels(x_px, y_px):
        ''' the item point drawn at (x_px, y_px) screen pixels from the origin '''
        return QtCore.QPointF(x_px / scale[0], y_px / scale[1])

    # an L-shaped track, 200 pixels along x, then 200 up in y
    line = polyLine([at_pixels(0, 0), at_pixels(200, 0), at_pixels(200, 200)])
    scene.addItem(line)

    shape = line.shape()
    assert shape.contains(at_pixels(100, HOVER_PIXELS - 2))
    assert not shape.contains(at_pixels(100, HOVER_PIXELS + 3))
    assert shape.contains(at_pixels(200 - (HOVER_PIXELS - 2), 100))
    assert not shape.contains(at_pixels(200 - (HOVER_PIXELS + 3), 100))
    # inside the L, away from both segments, is not part of the line
    assert not shape.contains(at_pixels(150, 50))


def serve_points(products, kind, pfp_kind):
    """ three points: T0-tagged, past the cathode (read out by TPC 0), plain """
    from titus.readers import Associations, Points, T0s
    products[(kind, 'pandora::Reco2')] = Points(id=[0, 1, 2], x=[-100., 10., -50.], y=[0., 0., 0.],
                                                z=[250., 250., 300.])
    products[(pfp_kind, 'pandora::Reco2')] = Associations(
        left=[0], right=[0], left_product=[0], right_product=[1], meta_index=[-1],
        product_tags=['pandora::Reco2', 'pandora::Reco2'])
    products[('pfp_t0s', 'pandora::Reco2')] = Associations(
        left=[0], right=[0], left_product=[0], right_product=[1], meta_index=[-1],
        product_tags=['pandora::Reco2', 'pandora::Reco2'])
    products[('t0s', 'pandora::Reco2')] = T0s(time=[-500000.], trigger_type=[3], id=[0])


def draw_points(cls, tpc_module, sbnd, corrected=True):
    tpc_module.correct_3d_objects = lambda: corrected
    drawer = cls(FakeGalleryInterface(), sbnd, tpc_module)
    drawer.set_producer('pandora::Reco2')
    drawer.analyze()
    drawer.drawObjects()
    return drawer


@pytest.mark.parametrize('corrected', [True, False])
def test_vertices(qtbot, products, sbnd, tpc_module, capsys, corrected):
    from titus.drawables import Vertex
    from titus.drawables.items import Marker
    serve_points(products, 'vertices', 'pfp_vertices')
    drawer = draw_points(Vertex, tpc_module, sbnd, corrected)
    assert 'Warning' not in capsys.readouterr().out
    markers = drawer._drawnObjects
    # one marker per vertex per view
    assert len(markers) == 3 * 3 and all(isinstance(m, Marker) for m in markers)
    tagged = [m for m in markers if m.toolTip().startswith('Vertex 0')]
    if corrected:
        assert all('T0 -500.0 us' in m.toolTip() for m in tagged)
        assert all(m.brush().style() == QtCore.Qt.NoBrush for m in tagged)
    else:
        assert all('drawn as stored' in m.toolTip() for m in markers)
        assert all(m.brush().style() != QtCore.Qt.NoBrush for m in markers)
    drawer.clearDrawnObjects()
    assert drawer._drawnObjects == []


def test_spacepoints(qtbot, products, sbnd, tpc_module, capsys):
    from titus.drawables import SpacePoint
    from titus.readers import Associations, Hits
    serve_points(products, 'spacepoints', 'pfp_spacepoints')
    # spacepoint 1 (past the cathode) has a hit in TPC 0
    products[('hits', 'gaushit::Reco1')] = Hits(
        channel=[0], cryostat=[0], tpc=[0], plane=[2], wire=[833], peak_time=[3000.], rms=[1.], integral=[1.],
        peak_amplitude=[1.], start_tick=[0], end_tick=[0])
    products[('spacepoint_hits', 'pandora::Reco2')] = Associations(
        left=[1], right=[0], left_product=[0], right_product=[1], meta_index=[-1],
        product_tags=['pandora::Reco2', 'gaushit::Reco1'])
    drawer = draw_points(SpacePoint, tpc_module, sbnd)
    assert 'Warning' not in capsys.readouterr().out
    # per view: plain points filled, the T0-shifted and the moved one hollow
    scatters = drawer._drawnObjects
    assert len(scatters) == 2 * 3
    assert sorted(len(s.data) for s in scatters) == [1, 1, 1, 2, 2, 2]
    stored = draw_points(SpacePoint, tpc_module, sbnd, corrected=False)
    assert len(stored._drawnObjects) == 3 and all(len(s.data) == 3 for s in stored._drawnObjects)
