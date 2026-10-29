"""
Items drawn over a wire view: the run/event label (data or simulation),
its hover outline, and the arrows at the wires drawn in the waveform dock.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

sys.modules.setdefault('ROOT', mock.MagicMock())
pytest.importorskip('PyQt5')
from PyQt5 import QtCore, QtWidgets  # noqa: E402

from titus.gallery_interface.gallery_interface import GalleryInterface  # noqa: E402
from titus.geometry import Geometry  # noqa: E402
from titus.gui.widgets import CONTROL_DOCK_MAX_WIDTH, MovableLabel, limit_dock_width  # noqa: E402
from titus.modules.tpc_module import _WIRE_COLOR_CYCLE, TpcModule, WireView  # noqa: E402


@pytest.fixture(scope='module')
def sbnd():
    return Geometry.load('sbnd')


def test_real_data_from_the_event_auxiliary(qtbot):
    gi = GalleryInterface()
    # nothing open: dated as data
    assert gi.is_real_data()
    gi._gallery_event_handle = SimpleNamespace(eventAuxiliary=lambda: SimpleNamespace(isRealData=lambda: False))
    assert not gi.is_real_data()


@pytest.fixture
def view(qtbot, sbnd):
    gi = mock.Mock(run=lambda: 18255, event=lambda: 42, is_real_data=lambda: False,
                   date=lambda include_time: 'March 1, 2025')
    view = WireView(sbnd, plane=2, cryostat=0, gallery_interface=gi)
    qtbot.addWidget(view)
    view.resize(800, 500)
    n_ticks = int(2 * sbnd.tRange() + sbnd.cathodeGap())
    view._item.setImage(np.zeros((sbnd.wRange(2), n_ticks), dtype=np.float32))
    view.show()
    qtbot.waitExposed(view)
    return view


def test_simulation_labelled_without_a_date(view):
    view.drawEventNumber()
    assert view._label.toPlainText() == 'RUN 18255, EVENT 42\nSBND Simulation'
    view._gi.is_real_data = lambda: True
    view.drawEventNumber()
    assert view._label.toPlainText() == 'RUN 18255, EVENT 42\nMarch 1, 2025'


def test_label_outlined_on_hover(qtbot):
    label = MovableLabel('RUN 1')
    plain = label.border
    label.hoverEnterEvent(None)
    assert label.border.color() == QtCore.Qt.yellow
    label.hoverLeaveEvent(None)
    assert label.border == plain


def arrows(view):
    return [(arrow.pos().x(), arrow.pos().y()) for arrow, _, _ in view._wire_marks]


def test_wire_arrows_in_the_white_space(view):
    height = view._item.height()
    # zoomed out: room below and above the image
    view._view.setRange(xRange=(-100, 2100), yRange=(-1000, height + 1000), padding=0)
    view.set_wire_marks([(400, 0, _WIRE_COLOR_CYCLE[0]), (1200, 1, _WIRE_COLOR_CYCLE[1])])
    # tips on the image's bottom edge (first TPC) and top edge (second)
    assert arrows(view) == [(400.5, 0), (1200.5, height)]

    # no room: moved in just enough to be seen whole
    view._view.setRange(xRange=(0, 2000), yRange=(0, height), padding=0)
    (_, low), (_, high) = arrows(view)
    assert 0 < low < 0.1 * height and 0.9 * height < high < height

    view.set_wire_marks([])
    assert view._wire_marks == [] and not any(
        type(item).__name__ == 'ArrowItem' for item in view._view.addedItems)


def test_arrows_follow_the_drawn_waveforms(qtbot):
    module = TpcModule.__new__(TpcModule)
    views = {(2, 0): mock.Mock(), (1, 0): mock.Mock()}
    module._wire_views = views
    module._drawWireOption = mock.Mock(isChecked=lambda: True)
    module._wirePlotItems = {(0, 400): {'idx': 0, 'view': views[(2, 0)]},
                             (1, 90): {'idx': 1, 'view': views[(2, 0)]}}
    module._refresh_wire_marks()
    views[(2, 0)].set_wire_marks.assert_called_with([(400, 0, _WIRE_COLOR_CYCLE[0]),
                                                     (90, 1, _WIRE_COLOR_CYCLE[1])])
    views[(1, 0)].set_wire_marks.assert_called_with([])
    # wire drawing off: no arrows
    module._drawWireOption = mock.Mock(isChecked=lambda: False)
    module._refresh_wire_marks()
    views[(2, 0)].set_wire_marks.assert_called_with([])


def test_control_docks_limited_in_width(qtbot):
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)
    dock = limit_dock_width(QtWidgets.QDockWidget('Controls', window))
    window.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    window.resize(1600, 800)
    window.show()
    window.resizeDocks([dock], [700], QtCore.Qt.Horizontal)
    assert dock.width() <= CONTROL_DOCK_MAX_WIDTH
