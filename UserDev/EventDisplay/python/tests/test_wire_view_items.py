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
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

from titus.gallery_interface.gallery_interface import GalleryInterface  # noqa: E402
from titus.geometry import Geometry  # noqa: E402
from titus.gui.widgets import CONTROL_DOCK_MAX_WIDTH, LineSplitter, MovableLabel, limit_dock_width  # noqa: E402
from titus.modules.tpc_module import _WIRE_COLOR_CYCLE, TpcModule, WireView  # noqa: E402


@pytest.fixture(scope='module')
def sbnd():
    return Geometry.load('sbnd')


def fake_event(real_data, seconds=1740830400):
    """ an event handle as gallery's, its time stamp the seconds (here 2025-03-01 12:00 UTC) """
    aux = SimpleNamespace(isRealData=lambda: real_data, run=lambda: 18255, event=lambda: 42,
                          time=lambda: SimpleNamespace(value=lambda: seconds << 32))
    return SimpleNamespace(eventAuxiliary=lambda: aux)


def test_real_data_from_the_event_auxiliary(qtbot):
    gi = GalleryInterface()
    # nothing open: dated as data
    assert gi.is_real_data()
    gi._gallery_event_handle = SimpleNamespace(eventAuxiliary=lambda: SimpleNamespace(isRealData=lambda: False))
    assert not gi.is_real_data()


def test_date_of_data_and_simulation(qtbot):
    gi = GalleryInterface()
    assert gi.date() == 'No event selected'
    gi._gallery_event_handle = fake_event(real_data=True)
    assert gi.date() == 'March 01, 2025 - 12:00:00 UTC'
    assert gi.date(include_time=False, detector='sbnd') == 'March 01, 2025'
    # simulation: no (meaningless) time stamp, wherever the date is shown
    gi._gallery_event_handle = fake_event(real_data=False)
    assert gi.date() == gi.date(include_time=False) == 'Simulation'
    assert gi.date(detector='sbnd') == 'SBND Simulation'


@pytest.fixture
def settings(tmp_path):
    return QtCore.QSettings(str(tmp_path / 'titus.ini'), QtCore.QSettings.IniFormat)


def make_view(qtbot, sbnd, gi=None, settings=None, plane=2):
    view = WireView(sbnd, plane=plane, cryostat=0, gallery_interface=gi or mock.Mock(), settings=settings)
    qtbot.addWidget(view)
    return view


@pytest.fixture
def view(qtbot, sbnd):
    gi = GalleryInterface()
    gi._gallery_event_handle = fake_event(real_data=False)
    view = make_view(qtbot, sbnd, gi)
    view.resize(800, 500)
    n_ticks = int(2 * sbnd.tRange() + sbnd.cathodeGap())
    view._item.setImage(np.zeros((sbnd.wRange(2), n_ticks), dtype=np.float32))
    view.show()
    qtbot.waitExposed(view)
    return view


def test_simulation_labelled_without_a_date(view):
    view.drawEventNumber()
    assert view._label.toPlainText() == 'RUN 18255, EVENT 42\nSBND Simulation'
    view._gi._gallery_event_handle = fake_event(real_data=True)
    view.drawEventNumber()
    assert view._label.toPlainText() == 'RUN 18255, EVENT 42\nMarch 01, 2025'


def test_event_label_can_be_hidden(view):
    view.toggleLogo(True)
    assert view._logo.scene() is view.scene() and view._label.scene() is view.scene()
    # e.g. for screenshots: the logo alone
    view.showEventLabel(False)
    assert view._logo.scene() is view.scene() and view._label.scene() is None
    view.showEventLabel(True)
    assert view._label.scene() is view.scene()
    # no logo, no label
    view.toggleLogo(False)
    assert view._label.scene() is None


def enter_levels(view, lower, upper):
    view._lowerLevel.setText(str(lower))
    view._upperLevel.setText(str(upper))
    view._upperLevel.returnPressed.emit()


def levels(view):
    return int(view._lowerLevel.text()), int(view._upperLevel.text())


def test_color_range_kept_between_sessions(qtbot, sbnd, settings):
    default = tuple(sbnd.getLevels(2))
    view = make_view(qtbot, sbnd, settings=settings)
    assert levels(view) == default
    enter_levels(view, -5, 300)
    assert tuple(view._item.levels) == (-5, 300)
    # the next session's view, and only this plane's
    assert levels(make_view(qtbot, sbnd, settings=settings)) == (-5, 300)
    assert levels(make_view(qtbot, sbnd, settings=settings, plane=1)) == tuple(sbnd.getLevels(1))
    # raw digits have their own range, around the pedestal
    view.drawingRawDigits(True)
    pedestal = sbnd.getPedestal(2)
    assert levels(view) == (default[0] + pedestal, default[1] + pedestal)
    enter_levels(view, 2000, 2100)
    view.drawingRawDigits(False)
    assert levels(view) == (-5, 300)
    view.drawingRawDigits(True)
    assert levels(view) == (2000, 2100)
    # Reset view: back to the default, in the next session too
    view.drawingRawDigits(False)
    view.restoreDefaults()
    assert levels(view) == default
    assert levels(make_view(qtbot, sbnd, settings=settings)) == default


def test_color_range_not_kept_without_settings(qtbot, sbnd):
    enter_levels(make_view(qtbot, sbnd), -5, 300)
    assert levels(make_view(qtbot, sbnd)) == tuple(sbnd.getLevels(2))


def test_aspect_ratio_kept_between_sessions(qtbot, settings):
    def module():
        module = TpcModule.__new__(TpcModule)
        module._settings = settings
        module._gm = mock.Mock(current_geom='sbnd')
        module._wire_views = {(2, 0): mock.Mock()}
        module._lockAspectRatio = QtWidgets.QCheckBox()
        module._lockAspectRatio.stateChanged.connect(lambda state: module.lock_aspect_ratio(state))
        module.restore_view_state()
        return module

    first = module()
    assert not first._lockAspectRatio.isChecked()
    first._lockAspectRatio.setChecked(True)
    first._wire_views[(2, 0)].lockRatio.assert_called_with(QtCore.Qt.Checked)
    second = module()
    assert second._lockAspectRatio.isChecked()
    second._wire_views[(2, 0)].lockRatio.assert_called_with(QtCore.Qt.Checked)


def test_restore_defaults_only_resets_the_tpc_page(qtbot, settings, monkeypatch):
    from titus.modules import tpc_module
    monkeypatch.setattr(QtCore, 'QSettings', lambda *args: settings)
    module = TpcModule(larsoft_module=None, geom_module=None)
    settings.setValue('MainWindow/geometry', 'kept')
    settings.setValue(tpc_module._SET_ASPECT_RATIO, True)
    module._label_event.setChecked(False)
    module._label_font_size.setValue(30)
    assert settings.value(tpc_module._SET_LABEL_EVENT, type=int) == QtCore.Qt.Unchecked
    module.restore_settings_defaults()
    assert module._label_event.isChecked() and module._label_font_size.value() == 16
    assert settings.value('MainWindow/geometry') == 'kept'
    assert settings.value(tpc_module._SET_ASPECT_RATIO, type=bool)


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


def test_views_screenshot_without_window_color(qtbot, sbnd):
    """ the TPC views' container: white, a thin line between the views """
    splitter = LineSplitter(QtCore.Qt.Vertical)
    qtbot.addWidget(splitter)
    for plane in (0, 1):
        # owned by the splitter
        view = WireView(sbnd, plane=plane, cryostat=0, gallery_interface=mock.Mock())
        splitter.addWidget(view.getWidgetAndLayout()[0])
    splitter.resize(400, 300)
    splitter.show()
    qtbot.waitExposed(splitter)
    image = splitter.grab().toImage()
    window = splitter.palette().color(QtGui.QPalette.Window).rgb()
    colors = {(x, y): image.pixel(x, y) for x in (0, 200, 399) for y in range(image.height())}
    assert window not in colors.values()
    # one line, across, in the handle
    handle = splitter.handle(1).geometry()
    line = [y for y in range(image.height()) if colors[(200, y)] == LineSplitter.line_color.rgb()]
    assert len(line) == 1 and handle.top() <= line[0] <= handle.bottom()
    assert all(colors[(x, line[0])] == LineSplitter.line_color.rgb() for x in (0, 399))
