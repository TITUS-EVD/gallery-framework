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
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

import titus.drawables.drawable as drawable_module  # noqa: E402
from titus.display import ViewLayout, hit_positions, track_correction, track_segments, track_t0s, wire_image  # noqa: E402
from titus.drawables import Hit, RawDigit, RecoWire, Track, recoChannelROI  # noqa: E402
from titus.drawables.hit import HitCollection  # noqa: E402
from titus.geometry import Geometry  # noqa: E402
from titus.readers import KINDS, DumpReader, ProductCache, ProductNotFound, RawDigits  # noqa: E402

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

    assert sum(len(c) for c in drawer._drawnObjects) == len(hits)
    # view U, TPC 0: box centered on the wire and on the peak, +- RMS
    view = tpc_module._wire_views[(0, 0)]
    (collection,) = [c for c in drawer._drawnObjects if c in view._view.addedItems]
    first = hits.on_plane(0, 0, 0)
    rects = [collection.rect(i) for i in range(len(collection))]
    rects = sorted((r.x(), r.center().y(), r.height()) for r in rects if r.center().y() < 3400)
    expected = sorted(zip(hits.wire[first] + 0.0, hits.peak_time[first].astype(float), 2. * hits.rms[first]))
    np.testing.assert_allclose(rects, expected, rtol=1e-5)

    drawer.clearDrawnObjects()
    assert drawer._drawnObjects == []
    assert collection not in view._view.addedItems


def test_hit_collection_paints_and_hovers(qtbot):
    tooltips = []
    collection = HitCollection([0, 10], [0, 20], [1, 1], [5, 5], [255, 128],
                               tooltip=lambda i: tooltips.append(i) or f'hit {i}')
    scene = QtWidgets.QGraphicsScene()
    scene.addItem(collection)
    assert collection.boundingRect() == QtCore.QRectF(0, 0, 11, 25)

    image = QtGui.QImage(11, 25, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor('white'))
    painter = QtGui.QPainter(image)
    scene.render(painter, QtCore.QRectF(0, 0, 11, 25), QtCore.QRectF(0, 0, 11, 25))
    painter.end()
    # opaque black, half transparent black, untouched white
    assert image.pixelColor(0, 2).value() == 0
    assert 100 < image.pixelColor(10, 22).value() < 160
    assert image.pixelColor(5, 10).value() == 255

    assert collection.index_at(10.5, 21) == 1 and collection.index_at(5, 5) is None
    collection._hover(1)
    assert collection.toolTip() == 'hit 1' and tooltips == [1]
    collection._hover(None)
    assert collection.toolTip() == ''


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


def test_larsoft_module_loads_no_services(qtbot, monkeypatch):
    """ the services packages aren't imported: any use of them fails """
    from titus.modules.larsoft_module import LArSoftModule
    monkeypatch.setitem(sys.modules, 'SBNDservices', None)
    lsm = LArSoftModule('sbnd')
    assert lsm.configured_for == 'sbnd' and not hasattr(lsm, 'load_services')
    with pytest.raises(RuntimeError, match='not a supported detector'):
        LArSoftModule('uboone')

def serve_reco2(products, entry, with_associations=True):
    dump = DumpReader(DATA / 'sbnd-reco2-pandora')
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


def test_vertices_on_real_data(qtbot, products, sbnd, tpc_module, capsys):
    """ the whole chain: vertex -> PFParticle -> T0, and -> spacepoints -> hits for its TPC """
    from titus.drawables import Vertex
    dump = DumpReader(DATA / 'sbnd-reco2-pandora')
    for kind, tag in [('vertices', 'pandora::Reco2'), ('pfp_vertices', 'pandora::Reco2'),
                      ('pfp_spacepoints', 'pandora::Reco2'), ('spacepoints', 'pandora::Reco2'),
                      ('spacepoint_hits', 'pandora::Reco2'), ('pfp_t0s', 'pandora::Reco2'),
                      ('t0s', 'pandora::Reco2'), ('hits', 'gaushit::Reco1')]:
        products[(kind, tag)] = dump.get(0, kind, tag)
    drawer = draw_points(Vertex, tpc_module, sbnd)
    assert 'Warning' not in capsys.readouterr().out
    tips = [m.toolTip() for m in drawer._drawnObjects]
    assert any('T0 ' in t for t in tips)
    assert any("drawn in its particle's TPC" in t for t in tips)


@pytest.mark.parametrize('corrected', [True, False])
def test_showers(qtbot, products, sbnd, tpc_module, capsys, corrected):
    """ showers without PFParticle associations: drawn without corrections """
    from PyQt5 import QtWidgets
    from titus.drawables import Shower
    products[('showers', 'pandoraShower::Reco2')] = make_showers()
    tpc_module.correct_3d_objects = lambda: corrected
    drawer = Shower(FakeGalleryInterface(), sbnd, tpc_module)
    drawer.set_producer('pandoraShower::Reco2')
    drawer.analyze()
    drawer.drawObjects()
    assert 'Warning' not in capsys.readouterr().out
    items = drawer._drawnObjects
    # the zero-length shower isn't drawn; the others once per view
    assert len(items) == 2 * 3 and all(isinstance(i, QtWidgets.QGraphicsPolygonItem) for i in items)
    # corrected or not, nothing moves them; the tooltip says which
    expected = 'corrected: nothing to change' if corrected else 'drawn as stored ("Correct 3D objects" is off)'
    assert all(expected in i.toolTip() for i in items)
    assert all(i.pen().style() == QtCore.Qt.SolidLine for i in items)
    drawer.clearDrawnObjects()
    assert drawer._drawnObjects == []


def make_showers():
    from titus.readers import Showers
    return Showers(id=[0, 1, 2], start_x=[-100., 100., 0.], start_y=[0., 0., 0.], start_z=[100., 250., 50.],
                   dir_x=[0., 0., 0.], dir_y=[0., 0.6, 0.], dir_z=[1., 0.8, 1.], length=[100., 50., 0.],
                   open_angle=[0.2, 0.1, 0.1], best_plane=[2, 2, 2])


def test_wire_view_unite_cathodes(qtbot, sbnd):
    """ the cut follows new images and t0, and clicks map back to the full image """
    pytest.importorskip('PIL')
    from titus.modules.tpc_module import WireView

    view = WireView(sbnd, plane=0, cryostat=0, gallery_interface=mock.Mock())
    qtbot.addWidget(view)
    layout = ViewLayout(sbnd, plane=0)
    image = np.random.default_rng(0).uniform(size=layout.shape).astype('f4')
    view.drawPlane(image)
    assert view.removed_ticks() == 0 and view._item.image.shape == image.shape

    view.uniteCathodes(True)
    n = view.removed_ticks()
    cathode = sbnd.anodeCathodeTicks(0)[0][1]
    assert n == int(sbnd.tRange() - cathode) > 0
    assert view._item.image.shape == (image.shape[0], image.shape[1] - 2 * n)
    # a new event's image is cut too
    view.drawPlane(image * 2)
    assert view._item.image.shape[1] == image.shape[1] - 2 * n
    np.testing.assert_array_equal(view._item.image[:, :100], 2 * image[:, :100])
    # the cut moves with t0
    view.t0slide(100)
    assert view.removed_ticks() == n - 100
    view.restoret0()
    assert view.removed_ticks() == n
    # shown y -> full image y, either side of the gap
    gap = int(sbnd.cathodeGap())
    assert view._fullImageY(10) == 10
    first_end = sbnd.tRange() - n
    assert view._fullImageY(first_end + 1) == first_end + 1 + n
    assert view._fullImageY(first_end + gap + 5) == first_end + gap + 5 + 2 * n

    view.uniteCathodes(False)
    assert view.removed_ticks() == 0 and view._item.image.shape == image.shape


# --- reading each product once, and the TPC module's redraw paths ---

class CachingGalleryInterface(FakeGalleryInterface):
    def __init__(self):
        super().__init__()
        self.product_cache = ProductCache()


def count_reads(monkeypatch):
    ''' (kind, tag) of every read_event_product call, through the products fixture '''
    reads = []
    read = drawable_module.read_event_product

    def counting(lib, ev, kind, tag, **kwargs):
        reads.append((kind, tag))
        return read(lib, ev, kind, tag, **kwargs)

    monkeypatch.setattr(drawable_module, 'read_event_product', counting)
    return reads


def test_drawables_share_products_read_once(qtbot, products, sbnd, tpc_module, monkeypatch, capsys):
    serve_reco2(products, 1)
    reads = count_reads(monkeypatch)
    gi = CachingGalleryInterface()
    tpc_module.correct_3d_objects = lambda: True
    hits = Hit(gi, sbnd, tpc_module)
    hits.set_producer('gaushit::Reco1')
    tracks = Track(gi, sbnd, tpc_module)
    tracks.set_producer('pandoraTrack::Reco2')
    for _ in range(2):
        # e.g. a redraw after a GUI change: nothing is read again
        for drawer in (hits, tracks):
            drawer.analyze()
            drawer.drawObjects()
    assert 'Warning' not in capsys.readouterr().out
    assert len(reads) == len(set(reads)) and ('hits', 'gaushit::Reco1') in reads
    assert tracks._drawnObjects and hits._drawnObjects

    gi.product_cache.new_event()
    hits.analyze()
    assert reads.count(('hits', 'gaushit::Reco1')) == 2


def test_removed_drawable_stops_receiving_signals(qtbot, products, sbnd):
    gi = FakeGalleryInterface()
    drawer = RecoWire(gi, sbnd)
    connected = gi.receivers(gi.fileChanged)
    drawer.disconnect_signals()
    assert gi.receivers(gi.fileChanged) == connected - 1
    drawer.disconnect_signals()  # twice is harmless


def test_wire_images_computed_once_per_event(qtbot, products, sbnd):
    products[('wires', 'simtpc2d:dnnsp:DetSim')] = DumpReader(DATA / 'sbnd-reco1-trimmed').wires(0, 'simtpc2d:dnnsp')
    drawer = RecoWire(FakeGalleryInterface(), sbnd)
    drawer.set_producer('simtpc2d:dnnsp:DetSim')
    drawer.analyze()
    image = drawer.getPlane(2, 0)
    assert drawer.getPlane(2, 0) is image
    drawer.analyze()
    assert drawer.getPlane(2, 0) is not image
    np.testing.assert_array_equal(drawer.getPlane(2, 0), image)


class FakeWireView:
    def __init__(self, plane, cryostat):
        self._plane, self._cryostat = plane, cryostat
        self.images, self.visible = [], None

    def plane(self):
        return self._plane

    def cryostat(self):
        return self._cryostat

    def drawPlane(self, image):
        self.images.append(image)

    def setVisible(self, visible):
        self.visible = visible

    def drawBlank(self):
        self.images.append(None)

    def setWrapperVisible(self, show):
        pass

    def toggleLogo(self, show):
        pass

    def toggleScale(self, show):
        pass


def bare_tpc_module(selected):
    from titus.modules.tpc_module import TpcModule
    module = TpcModule.__new__(TpcModule)
    module._wire_views = {(p, 0): FakeWireView(p, 0) for p in range(3)}
    module._selected_planes = [p for p, _ in selected]
    module._selected_cryos = [c for _, c in selected]
    module._stale_views = set()
    module._draw_wires = True
    module._wire_drawer = mock.Mock(getPlane=lambda plane, cryo: f'image {plane}')
    module._plane_frames = mock.Mock(isChecked=lambda: True)
    module._current_wire_drawer = None
    module._show_logo = module._show_scale_bar = False
    module._product_views = set()
    module._product_box_map = {'box': mock.Mock()}
    return module


def test_hidden_views_drawn_when_shown():
    module = bare_tpc_module([(2, 0)])
    module.update()
    views = module._wire_views
    assert views[(2, 0)].images == ['image 2']
    assert views[(0, 0)].images == [] and module._stale_views == {(0, 0), (1, 0)}

    module._selected_planes, module._selected_cryos = [-1], [-1]
    module.refresh_draw_list_widget()
    assert [v.images for v in views.values()] == [['image 0'], ['image 1'], ['image 2']]
    assert all(v.visible for v in views.values()) and not module._stale_views
    # products were drawn in the collection view only: drawn again, now in all three
    products = module._product_box_map['box']
    assert products.drawObjects.call_count == 1 and module._product_views == set(views)
    # already drawn for this event
    module.refresh_draw_list_widget()
    assert views[(0, 0)].images == ['image 0'] and products.drawObjects.call_count == 1


def test_no_wires_blanks_shown_views():
    module = bare_tpc_module([(-1, -1)])
    module._draw_wires = False
    module.update()
    assert [v.images for v in module._wire_views.values()] == [[None]] * 3


def test_icarus_draws_every_drift_volume_product():
    from titus.gallery_interface.gallery_interface import Product
    from titus.modules.tpc_module import TpcModule
    products = [Product(f'recobChannelROIs_wire2channelroi2d_PHYSCRATEDATATPC{tpc}_stage0', 'std::vector<recob::ChannelROI>')
                for tpc in ('EE', 'EW', 'WE', 'WW')]
    products.append(Product('recobChannelROIs_other_PHYSCRATEDATATPCEE_stage1', 'std::vector<recob::ChannelROI>'))
    box = mock.Mock(selected_products=lambda: ['wire2channelroi2d:PHYSCRATEDATATPCEW:stage0'])
    module = TpcModule.__new__(TpcModule)
    module._gm = mock.Mock()
    module._gm.current_geom.name.return_value = 'icarus'
    assert module._wire_producers(box, products) == [p.full_name() for p in products[:4]]
    assert module._wire_producers(box, products, producers=['given']) == ['given']
    module._gm.current_geom.name.return_value = 'sbnd'
    assert module._wire_producers(box, products) == 'wire2channelroi2d:PHYSCRATEDATATPCEW:stage0'


def test_crt_update_without_crt_products():
    from titus.modules.crt_module import CrtModule
    module = CrtModule.__new__(CrtModule)
    module._gi = mock.Mock(get_producers=lambda *args: None)
    module._lsm = mock.Mock()
    module._crt_strip_drawer = None
    module.update()
    assert module._draw_crt_strips is False


STEP3 = DATA / 'sbnd-reco2-step3'


def serve_all(products, path, entry=0):
    ''' every product of a dump event, by full tag '''
    reader = DumpReader(path)
    for kind in KINDS:
        for tag in reader.tags(kind, entry):
            products[(kind, tag)] = reader.get(entry, kind, tag)
    return reader


def test_showers_on_real_data(qtbot, products, sbnd, tpc_module, capsys):
    ''' Pandora showers with their particles: corrections found, every shower with a length drawn '''
    from titus.drawables import Shower
    reader = serve_all(products, STEP3)
    showers = reader.get(0, 'showers', 'pandoraShower')
    tpc_module.correct_3d_objects = lambda: True
    drawer = Shower(FakeGalleryInterface(), sbnd, tpc_module)
    drawer.set_producer('pandoraShower::Reco2')
    drawer.analyze()
    drawer.drawObjects()
    assert 'Warning' not in capsys.readouterr().out
    tips = [i.toolTip() for i in drawer._drawnObjects]
    drawn = {t.split(':')[0] for t in tips}
    assert drawn == {f'Shower {i}' for i, n in zip(showers.id, showers.length) if n > 0}
    # the cosmic showers carry a T0, and every tooltip says what the corrections did
    assert any('T0 ' in t for t in tips)
    assert not any('Correct 3D objects' in t for t in tips)


def test_clusters_drawn_as_their_hits(qtbot, products, sbnd, tpc_module, capsys):
    from titus.drawables import Cluster
    reader = serve_all(products, STEP3)
    drawer = Cluster(FakeGalleryInterface(), sbnd, tpc_module)
    drawer.set_producer('pandora::Reco2')
    drawer.analyze()
    drawer.drawObjects()
    assert 'Warning' not in capsys.readouterr().out
    assns = reader.get(0, 'cluster_hits', 'pandora')
    # every associated hit, once per view of its plane
    assert sum(len(c) for c in drawer._drawnObjects) == len(assns)
    (collection,) = [c for c in drawer._drawnObjects if c in tpc_module._wire_views[(2, 0)]._view.addedItems]
    # hovering a hit outlines its cluster: the cluster's hits on this view
    collection._hover(0)
    cluster = collection.group[0]
    assert len(collection.highlighted()) == np.sum(collection.group == cluster) > 1
    clusters = reader.get(0, 'clusters', 'pandora')
    assert collection.toolTip().startswith(f'Cluster {clusters.id[cluster]}:')
    # a color per cluster
    assert len({tuple(c) for c in collection.colors}) > 1
    drawer.clearDrawnObjects()
    assert drawer._drawnObjects == []


def test_clusters_without_associations(qtbot, products, sbnd, tpc_module, capsys):
    from titus.drawables import Cluster
    products[('clusters', 'pandora::Reco2')] = DumpReader(STEP3).get(0, 'clusters', 'pandora')
    drawer = Cluster(FakeGalleryInterface(), sbnd, tpc_module)
    drawer.set_producer('pandora::Reco2')
    drawer.analyze()
    drawer.drawObjects()
    assert 'no cluster-hit associations' in capsys.readouterr().out
    assert drawer._drawnObjects == []


def test_endpoints2d(qtbot, products, sbnd, tpc_module):
    from titus.drawables import EndPoint2D
    from titus.drawables.items import Marker
    from titus.readers import EndPoints2D
    products[('endpoints2d', 'ep::Reco1')] = EndPoints2D(
        id=[0, 1], cryostat=[0, 0], tpc=[0, 1], plane=[2, 2], wire=[100, 200], drift_time=[1000., 2000.],
        charge=[1., 1.], strength=[3, 4])
    drawer = EndPoint2D(FakeGalleryInterface(), sbnd, tpc_module)
    drawer.set_producer('ep::Reco1')
    drawer.analyze()
    drawer.drawObjects()
    # both on the collection view, at the same place as a hit on that wire and time
    markers = drawer._drawnObjects
    assert len(markers) == 2 and all(isinstance(m, Marker) for m in markers)
    assert all(m in tpc_module._wire_views[(2, 0)]._view.addedItems for m in markers)
    layout = ViewLayout(sbnd, plane=2)
    plane = sbnd.getPlaneID(plane=2, tpc=1, cryo=0)
    assert markers[1].pos().y() == pytest.approx(layout.tick_to_view(plane, 2000. + sbnd.timeOffsetTicks(2)))


def test_mctracks_corrected_by_their_time(qtbot, products, sbnd, tpc_module, capsys):
    from titus.drawables import MCTrack
    serve_all(products, STEP3)
    tips = {}
    for corrected in (True, False):
        tpc_module.correct_3d_objects = lambda corrected=corrected: corrected
        drawer = MCTrack(FakeGalleryInterface(), sbnd, tpc_module)
        drawer.set_producer('mcreco::G4')
        drawer.analyze()
        drawer.drawObjects()
        tips[corrected] = [l.toolTip() for l in drawer._drawnObjects]
        assert drawer._drawnObjects
    assert 'Warning' not in capsys.readouterr().out
    assert any('T0 ' in t for t in tips[True]) and all('Correct 3D objects' in t for t in tips[False])
    # neutrino particles gray, cosmics black
    colors = {l.pen().color().getRgb()[:3] for l in drawer._drawnObjects}
    assert (0, 0, 0) in colors


def test_mctrack_time_puts_cosmics_on_their_hits(sbnd):
    ''' out-of-time cosmic muons: corrected by their MC time, their steps lie on the collection hits '''
    from titus.display import mc_t0, track_correction, track_segments
    from titus.drawables.mctrack import mctracks_as_tracks
    reader = DumpReader(STEP3)
    mctracks = reader.get(0, 'mctracks', 'mcreco')
    tracks = mctracks_as_tracks(mctracks)
    layout = ViewLayout(sbnd, plane=2)
    positions = hit_positions(layout, reader.hits(0, 'gaushit'), sbnd.timeOffsetTicks(2))
    hit_ticks = {}
    for x, y in zip(np.floor(positions.x).astype(int), positions.y):
        hit_ticks.setdefault(x, []).append(y)
    t0 = mc_t0(sbnd, mctracks.start_t)
    long_rows = [i for i in range(len(mctracks))
                 if len(tracks.points(i)) > 1 and np.linalg.norm(np.ptp(tracks.points(i), axis=0)) > 100]
    assert long_rows and all(abs(t0[i]) > 5e3 for i in long_rows)  # this event's long tracks are out of time

    def median_distance(correction_of):
        d = [min(abs(np.array(hit_ticks[int(np.floor(x))]) - y))
             for i in long_rows for s in track_segments(layout, tracks, i, correction_of(i))
             for x, y in zip(s.x, s.y) if int(np.floor(x)) in hit_ticks]
        return np.median(d)

    assert median_distance(lambda i: track_correction(layout, tracks, i, t0[i])) < 20
    assert median_distance(lambda i: None) > 500


@pytest.fixture
def truth_module(tpc_module):
    ''' the TPC module's MC truth panel; dock_shown records show_mctruth_dock '''
    from titus.gui.mctruth_panel import MCTruthPanel
    tpc_module._mctruth_panel = MCTruthPanel()
    tpc_module.dock_shown = []
    tpc_module.show_mctruth_dock = tpc_module.dock_shown.append
    return tpc_module


def draw_truth(sbnd, module, producers=('generator::GenieGen', 'corsika::GenieGen')):
    from titus.drawables import MCTruth
    module.correct_3d_objects = lambda: True
    drawer = MCTruth(FakeGalleryInterface(), sbnd, module)
    drawer.set_producer(list(producers))
    drawer.analyze()
    drawer.drawObjects()
    return drawer


def test_mctruth_neutrino_first_with_vertex(qtbot, products, sbnd, truth_module, capsys):
    from titus.drawables.items import CrossMarker
    reader = serve_all(products, STEP3)
    drawer = draw_truth(sbnd, truth_module)
    assert 'Warning' not in capsys.readouterr().out
    panel = truth_module._mctruth_panel
    assert [panel.dropdown.itemText(i) for i in range(2)] == ['νμ NC (generator 1)', 'Cosmic (corsika 1)']
    summary = ' '.join(f'{label} {value}' for label, value in panel.summary())
    for expected in ('νμ (14), 1.526 GeV', 'NC RES (type 1006)', 'Ar40', 'TPC 0', 'Final state'):
        assert expected in summary
    assert truth_module.dock_shown[-1] is True
    # the final state, most energetic first; all particles with their GENIE status
    truths = reader.get(0, 'mctruths', 'generator')
    pdg, status, _ = truths.particles(0)
    assert panel.particle_count() == (status == 1).sum() == 11
    kinetic = [float(panel.cell(r, 'KE [MeV]')) for r in range(panel.particle_count())]
    assert kinetic == sorted(kinetic, reverse=True)
    panel.all_particles.setChecked(True)
    assert panel.particle_count() == len(pdg) == 21
    assert 'hadron in nucleus' in [panel.cell(r, 'Status') for r in range(21)]

    crosses = drawer._drawnObjects
    assert crosses and all(isinstance(c, CrossMarker) for c in crosses)
    assert crosses[0].toolTip().startswith('νμ interaction vertex')
    panel.show_vertex.setChecked(False)
    assert not any(c.isVisible() for c in crosses)

    # the cosmics: described and listed; the neutrino's vertex stays, not highlighted
    panel.all_particles.setChecked(False)
    panel.dropdown.setCurrentIndex(1)
    assert ('Origin', 'Cosmic') in panel.summary() and drawer._drawnObjects == crosses
    assert not any(c.is_selected() for c in crosses) and panel.show_vertex.isEnabled()
    assert panel.particle_count() == len(reader.get(0, 'mctruths', 'corsika').particles(0)[0])

    # no MC truth producer: the dock is empty and hidden
    drawer.set_producer(None)
    assert truth_module.dock_shown[-1] is False and panel.particle_count() == 0


def test_mctruth_dock_hidden_without_truth_in_the_event(qtbot, products, sbnd, truth_module):
    from titus.readers import MCTruths
    products[('mctruths', 'generator::GenieGen')] = MCTruths.empty()
    draw_truth(sbnd, truth_module, producers=['generator::GenieGen'])
    assert truth_module.dock_shown == [False]
    assert truth_module._mctruth_panel.dropdown.count() == 0


# --- optical and CRT ---

class FakeOpdetModule:
    def __init__(self):
        self.flashes = {}

    def setFlashesForPlane(self, p, flashes):
        self.flashes[p] = flashes


def test_flashes_assigned_to_their_tpc(qtbot, products, sbnd):
    ''' SBND makes flashes per TPC (opflashtpc0/1): each lands on its drift volume '''
    from titus.drawables import OpFlash
    reader = serve_all(products, STEP3)
    for volume in (0, 1):
        module = FakeOpdetModule()
        drawer = OpFlash(FakeGalleryInterface(), sbnd, module)
        drawer.set_producer(f'opflashtpc{volume}::Reco1')
        drawer.analyze()
        drawer.drawObjects()
        flashes = reader.get(0, 'opflashes', f'opflashtpc{volume}')
        assert len(flashes)
        assert len(module.flashes[volume]) == len(flashes) and module.flashes.get(1 - volume) is None
        first = module.flashes[volume][0]
        assert first.time() == flashes.abs_time[0]
        assert len(first.pe_per_opdet()) == len(sbnd.detector.opdets)
        drawer.clearDrawnObjects()
        assert module.flashes == {0: None, 1: None}


def make_febs(flags, ts0, coinc, unix_s, adc_max_at, adc_max=500):
    from titus.readers import CRTFEBData
    n = len(flags)
    adc = np.full((n, 32), 10)
    adc[np.arange(n), adc_max_at] = adc_max
    return CRTFEBData(mac5=np.arange(n) + 7, flags=flags, ts0=ts0, ts1=np.full(n, 99), coinc=coinc,
                      unix_s=unix_s, adc=adc)


def test_crt_strip_hits_without_trigger():
    from titus.display.crt import crt_strip_hits
    febs = make_febs(flags=[1, 3, 7], ts0=[1000, 2000, 3000], coinc=[5, 6, 7], unix_s=[0, 0, 0], adc_max_at=[3, 31, 0])
    hits = crt_strip_hits(febs)
    # the reset (flags 7) is dropped; one row per readout, for its largest SiPM
    assert hits.tolist() == [[32 * 7 + 3, 1005, 99, 500], [32 * 8 + 31, 2006, 99, 500]]


def test_crt_strip_hits_relative_to_the_trigger():
    from titus.display.crt import crt_strip_hits
    from titus.readers import DAQTimestamps
    trigger = 1_700_000_000 * 10 ** 9 + 999_999_000
    stamps = DAQTimestamps(channel=[1, 4], timestamp=[5, trigger])
    febs = make_febs(flags=[1, 1, 1], ts0=[999_998_000, 1_000, 999_999_500], coinc=[0, 0, 0],
                     unix_s=[1_700_000_000, 1_700_000_001, 1_699_999_999], adc_max_at=[0, 0, 0])
    t0 = crt_strip_hits(febs, stamps)[:, 1]
    # same second; the next second (+1 s); the second before (-1 s)
    assert t0.tolist() == [-1000, 1_000 + 10 ** 9 - 999_999_000, 999_999_500 - 10 ** 9 - 999_999_000]
    with pytest.raises(ValueError, match='more than 1 s'):
        crt_strip_hits(make_febs([1], [0], [0], [1_700_000_005], [0]), stamps)


def test_febdata_drawable(qtbot, products, sbnd):
    from titus.drawables import FEBData
    reader = serve_all(products, STEP3)
    drawer = FEBData(FakeGalleryInterface(), sbnd)
    drawer.set_producer('crtsim::DetSim')
    drawer.analyze()
    hits = drawer.getData()
    febs = reader.get(0, 'crt_febdata', 'crtsim')
    assert hits.shape == (np.isin(febs.flags, (1, 3)).sum(), 4) and len(hits)
    # MC: no trigger time stamp, times as stored
    keep = np.isin(febs.flags, (1, 3))
    assert hits[:, 1].tolist() == (febs.ts0[keep].astype(np.int64) + febs.coinc[keep]).tolist()


class SteppingEvent:
    ''' a gallery.Event of n entries, logging moves; goToEntry only if direct '''
    def __init__(self, n, direct):
        self.n, self.entry, self.log = n, 0, []
        if direct:
            self.goToEntry = lambda e: (self.log.append(('goToEntry', e)), setattr(self, 'entry', e))

    def atEnd(self):
        return self.entry >= self.n

    def eventEntry(self):
        return self.entry

    def toBegin(self):
        self.log.append('toBegin')
        self.entry = 0

    def next(self):
        self.log.append('next')
        self.entry += 1


@pytest.mark.parametrize('direct', [True, False])
def test_position_event(direct):
    from titus.gallery_interface.gallery_interface import position_event
    ev = SteppingEvent(10, direct)
    for entry in (5, 6, 2):
        position_event(ev, entry)
        assert ev.eventEntry() == entry
    if direct:
        assert ev.log == [('goToEntry', 5), ('goToEntry', 6), ('goToEntry', 2)]
    else:
        # back from 6 to 2: rewind and step
        assert ev.log == ['next'] * 6 + ['toBegin', 'next', 'next']
    ev.entry = 10  # past the end
    position_event(ev, 3)
    assert ev.eventEntry() == 3


def test_products_drawn_in_shown_views_only(qtbot, products, sbnd, tpc_module):
    products[('hits', 'gaushit::Reco1')] = DumpReader(DATA / 'sbnd-reco1-trimmed').hits(0, 'gaushit')
    tpc_module._view_shown = lambda key: key == (2, 0)
    drawer = Hit(FakeGalleryInterface(), sbnd, tpc_module)
    drawer.set_producer('gaushit::Reco1')
    drawer.analyze()
    drawer.drawObjects()
    (collection,) = drawer._drawnObjects
    assert collection in tpc_module._wire_views[(2, 0)]._view.addedItems


CRT_DATA = DATA / 'sbnd-data-crt'


def test_crt_times_relative_to_the_trigger_on_data(qtbot, products, sbnd):
    ''' SBND data: FEB times are since the second (PPS); the drawable finds tdcdecoder and refers them to the trigger '''
    from titus.drawables import FEBData
    serve_all(products, CRT_DATA)
    drawer = FEBData(FakeGalleryInterface(), sbnd)
    drawer.set_producer('crtdecoder::DECODE')
    drawer.analyze()
    t0 = drawer.getData()[:, 1]
    assert len(t0) > 1000
    # the CRT readout spans tens of ms around the trigger
    assert t0.min() < 0 < t0.max() and np.all(np.abs(t0) < 30e6) and abs(np.median(t0)) < 5e6
    reader = DumpReader(CRT_DATA)
    from titus.display.crt import crt_strip_hits
    assert np.all(crt_strip_hits(reader.get(0, 'crt_febdata', 'crtdecoder'))[:, 1] > 100e6)


def make_waveforms(rows):
    ''' rows: (channel, time stamp, samples) '''
    from titus.readers import OpDetWaveforms
    samples = [np.asarray(s) for _, _, s in rows]
    return OpDetWaveforms(channel=[r[0] for r in rows], time_stamp=[r[1] for r in rows],
                          adc_offset=np.concatenate([[0], np.cumsum([len(s) for s in samples])]),
                          adc=np.concatenate(samples) if samples else [])


def test_opdet_traces():
    from titus.display.optical import opdet_traces
    waveforms = make_waveforms([(3, 10., [1, 2, 3, 4, 5]), (3, 0., [7, 8]), (900, 0., [1]), (5, 1., [9])])
    traces = opdet_traces(waveforms, n_channels=312, tick_period=0.5, compression=2, time_offset=100.)
    # the timing channel isn't an optical detector
    assert sorted(traces) == [3, 5]
    # channel 3: both waveforms, earliest first, every second sample, nan between
    np.testing.assert_array_equal(traces[3]['adc'], [7, np.nan, 1, 3, 5, np.nan])
    np.testing.assert_array_equal(traces[3]['time'], [100., np.nan, 110., 111., 112., np.nan])
    assert opdet_traces(make_waveforms([]), 312, 0.5) == {}


def test_opdetwaveform_drawable(qtbot, products, sbnd):
    from titus.drawables import OpDetWaveform
    products[('opdet_waveforms', 'pmtdecoder:PMTChannels:DECODE')] = make_waveforms([(6, -2., np.arange(100))])
    drawer = OpDetWaveform(FakeGalleryInterface(), sbnd)
    drawer.set_producer('pmtdecoder:PMTChannels:DECODE')
    drawer.analyze()
    assert len(drawer.getData()[6]['adc']) == 10 + 1  # default compression 10, and the nan
    drawer.set_compression_factor(1)
    assert len(drawer.getData()[6]['adc']) == 100 + 1


def test_truth_dumper_lines():
    from titus.modules.truthdumper_module import TruthPrintOut
    truths = DumpReader(STEP3).get(0, 'mctruths', 'generator')
    (line,) = TruthPrintOut.lines(truths)
    assert line.startswith(TruthPrintOut.NU_NAME_MAP.get(int(truths.nu_pdg[0]), ''))
    assert TruthPrintOut.lines(DumpReader(STEP3).get(0, 'mctruths', 'corsika')) == []


def test_modules_import_without_header_parsing(monkeypatch):
    ''' importing the modules doesn't use galleryUtils (it made cling parse MCTruth.h from cvmfs at startup) '''
    import importlib
    monkeypatch.setitem(sys.modules, 'galleryUtils', None)
    import titus.modules.truthdumper_module as truthdumper
    importlib.reload(truthdumper)


def test_product_box_choice_for_a_new_file():
    from titus.modules.tpc_module import box_choice
    # MC files show the generator's truth unless another producer was chosen
    assert box_choice('simb::MCTruth', None, ['corsika', 'generator']) == 'generator'
    assert box_choice('simb::MCTruth', 'corsika', ['corsika', 'generator']) == 'corsika'
    assert box_choice('simb::MCTruth', None, ['corsika']) is None
    # other products: the previous choice, if the new file has it
    assert box_choice('recob::Hit', 'gaushit', ['gaushit', 'fasthit']) == 'gaushit'
    assert box_choice('recob::Hit', 'gaushit', ['fasthit']) is None
    assert box_choice('recob::Hit', None, ['gaushit']) is None


def test_mctruth_dock_follows_the_module(qtbot):
    from titus.modules.tpc_module import TpcModule
    module = TpcModule.__new__(TpcModule)
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)
    module._mctruth_dock = QtWidgets.QDockWidget('MC Truth', window)
    window.addDockWidget(QtCore.Qt.RightDockWidgetArea, module._mctruth_dock)
    window.show()
    module._dock_widgets, module._active = set(), True
    module.init_mctruth()
    assert module._mctruth_dock.isHidden() and module._mctruth_dropdown is module._mctruth_panel.dropdown
    module.show_mctruth_dock(True)
    assert module._mctruth_dock.isVisible() and module._mctruth_dock in module._dock_widgets
    module.show_mctruth_dock(False)
    assert module._mctruth_dock.isHidden() and module._mctruth_dock not in module._dock_widgets
    # while another view is shown: only once the TPC view is back (Module.activate shows _dock_widgets)
    module._active = False
    module.show_mctruth_dock(True)
    assert module._mctruth_dock.isHidden() and module._mctruth_dock in module._dock_widgets


def test_all_neutrino_vertices_the_selected_highlighted(qtbot, products, sbnd, truth_module):
    ''' two neutrino interactions: both vertices drawn, the selected one highlighted; one toggle for all '''
    from titus.readers import MCTruths
    truths = DumpReader(STEP3).get(0, 'mctruths', 'generator')
    arrays = truths.to_arrays()
    # a second interaction: the same one, moved 100 cm in z
    two = {k: np.concatenate([v, v]) for k, v in arrays.items() if not k.startswith('particle_')}
    two['vz'] = two['vz'] + np.array([0., 100.], dtype='f4')
    n = len(truths.particle_pdg)
    two.update(particle_offset=[0, n, 2 * n], **{k: np.concatenate([arrays[k], arrays[k]])
                                                  for k in ('particle_pdg', 'particle_status', 'particle_energy')})
    products[('mctruths', 'generator::GenieGen')] = MCTruths(**two)
    drawer = draw_truth(sbnd, truth_module, producers=['generator::GenieGen'])
    panel = truth_module._mctruth_panel
    crosses = drawer._crosses
    # one cross per interaction per view (3 SBND views)
    assert sorted(c for c, _ in crosses) == [0, 0, 0, 1, 1, 1]
    assert all(cross.is_selected() == (choice == 0) for choice, cross in crosses)
    panel.dropdown.setCurrentIndex(1)
    assert all(cross.is_selected() == (choice == 1) for choice, cross in crosses)
    assert drawer._crosses == crosses  # restyled, not redrawn
    panel.show_vertex.setChecked(False)
    assert not any(cross.isVisible() for _, cross in crosses)
    panel.show_vertex.setChecked(True)
    assert all(cross.isVisible() for _, cross in crosses)


def test_selected_particles_highlight_their_mctracks(qtbot, products, sbnd, truth_module):
    from titus.drawables import MCTrack
    from titus.display.mctruth import particle_key
    from titus.modules.tpc_module import TpcModule
    reader = serve_all(products, STEP3)
    # the TPC module's selection, with its product boxes
    truth_module.select_mc_particles = lambda keys: TpcModule.select_mc_particles(truth_module, keys)
    truth_module.selected_mc_particles = lambda: TpcModule.selected_mc_particles(truth_module)
    truth_module._selected_mc_particles = []
    truth_module._product_box_map = {}
    drawer = draw_truth(sbnd, truth_module, producers=['generator::GenieGen'])
    panel = truth_module._mctruth_panel
    truths = reader.get(0, 'mctruths', 'generator')
    pdg, status, energy = truths.particles(0)
    protons = [i for i in range(len(pdg)) if status[i] == 1 and pdg[i] == 2212]
    neutron = next(i for i in range(len(pdg)) if status[i] == 1 and pdg[i] == 2112)

    # no MCTracks drawn: the dock says how to see them
    panel.select_particles([protons[0]])
    assert truth_module.selected_mc_particles() == [particle_key(truths, 0, protons[0])]
    assert panel.track_note().startswith('Turn on MCTrack')

    tracks = MCTrack(FakeGalleryInterface(), sbnd, truth_module)
    tracks.set_producer('mcreco::G4')
    truth_module._product_box_map = {'MCTrack box': tracks}
    tracks.analyze()
    tracks.drawObjects()
    # drawn after the selection: highlighted already
    highlighted = {l.track_key for l in tracks._drawnObjects if l.is_selected()}
    assert len(highlighted) == 1
    (producer, row), = highlighted
    assert tracks._mctracks[producer].process[row] == 'primary'
    assert tracks._mctracks[producer].start_e[row] == pytest.approx(energy[protons[0]] * 1000, abs=0.05)

    # two protons and a neutron: two tracks, and a note on the neutron
    panel.select_particles(protons[:2] + [neutron])
    assert len({l.track_key for l in tracks._drawnObjects if l.is_selected()}) == 2
    note = panel.track_note()
    assert note.count('p: MCTrack 1000000') == 2 and 'No MCTrack for n' in note
    # another interaction or event: nothing selected
    panel.dropdown.setCurrentIndex(0)
    drawer.on_selection_changed(0)
    assert truth_module.selected_mc_particles() == [] and not any(l.is_selected() for l in tracks._drawnObjects)


def test_hover_follows_the_line_in_a_view(qtbot):
    ''' in a pyqtgraph ViewBox (which clips its items): hover and tooltips only near the line, not its bounding box '''
    from titus.drawables.items import polyLine
    widget = pg.GraphicsLayoutWidget()
    qtbot.addWidget(widget)
    widget.resize(600, 600)
    view = widget.addViewBox()
    view.setRange(xRange=(0, 100), yRange=(0, 100), padding=0)
    line = polyLine([QtCore.QPointF(0, 0), QtCore.QPointF(100, 100)], (0, 0, 0))
    view.addItem(line)
    widget.show()
    qtbot.waitExposed(widget)
    scene = widget.scene()

    def hovered(x, y):
        ''' what Qt finds under the point (view coordinates): hover and tooltips use this '''
        return line in scene.items(view.mapViewToScene(QtCore.QPointF(x, y)))

    # ~6 pixels per unit: on the line, 0.5 units (3 px) off, 2 units (12 px) off, far off inside the bounding box
    assert hovered(50, 50) and hovered(50, 50.5)
    assert not hovered(50, 52) and not hovered(20, 80)
    # the hover distance is in screen pixels, at any zoom
    view.setRange(xRange=(40, 60), yRange=(40, 60), padding=0)
    QtWidgets.QApplication.processEvents()
    assert hovered(50, 50.1) and not hovered(50, 50.5)
    # still painted whole: the clip (the shape) is wider than the pen
    image = widget.grab().toImage()
    center = view.mapViewToScene(QtCore.QPointF(50, 50))
    assert QtGui.QColor(image.pixel(int(center.x()), int(center.y()))).value() < 100


def test_shower_hover_follows_its_cone(qtbot):
    from titus.drawables.items import hover_on_shape
    widget = pg.GraphicsLayoutWidget()
    qtbot.addWidget(widget)
    widget.resize(600, 600)
    view = widget.addViewBox()
    view.setRange(xRange=(0, 100), yRange=(0, 100), padding=0)
    # a thin cone along the diagonal
    cone = hover_on_shape(QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF(
        [QtCore.QPointF(0, 0), QtCore.QPointF(100, 95), QtCore.QPointF(95, 100)])))
    view.addItem(cone)
    widget.show()
    qtbot.waitExposed(widget)
    scene = widget.scene()
    assert cone in scene.items(view.mapViewToScene(QtCore.QPointF(60, 60)))
    assert cone not in scene.items(view.mapViewToScene(QtCore.QPointF(20, 80)))
