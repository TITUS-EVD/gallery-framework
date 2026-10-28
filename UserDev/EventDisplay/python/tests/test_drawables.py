"""
Wire and hit drawables on the reader path, offscreen, with products from the
test dumps in place of a gallery event. The other drawables still import the
C++ evd library, so ROOT is stubbed to import titus.drawables.
"""

import pathlib
import sys
from unittest import mock

import numpy as np
import pytest

sys.modules.setdefault('ROOT', mock.MagicMock())

pg = pytest.importorskip('pyqtgraph')
from PyQt5 import QtCore  # noqa: E402

import titus.drawables.drawable as drawable_module  # noqa: E402
from titus.display import ViewLayout, wire_image  # noqa: E402
from titus.drawables import Hit, RawDigit, RecoWire, recoChannelROI  # noqa: E402
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

    def read(lib, ev, kind, tag, columns=None):
        if (kind, tag) not in served:
            raise ProductNotFound(f'{kind} {tag} not in this event')
        return served[(kind, tag)]

    monkeypatch.setattr(drawable_module, 'read_event_product', read)
    monkeypatch.setattr(drawable_module, 'titusio', lambda: None)
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
