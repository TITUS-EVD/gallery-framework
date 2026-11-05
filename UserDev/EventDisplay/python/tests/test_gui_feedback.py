"""
What the GUI shows while it is busy (loading signals, overlay, progress
bar), product boxes without products, and the wire view chosen for a new
file. gallery is faked.
"""

import sys
from unittest import mock

import pytest

sys.modules.setdefault('ROOT', mock.MagicMock())
pytest.importorskip('PyQt5')
from PyQt5 import QtCore, QtWidgets  # noqa: E402

from titus.gallery_interface.gallery_interface import GalleryInterface, OpenedFiles, OpenError, product_table  # noqa: E402
from titus.gui.widgets import MultiSelectionBox, recoBox  # noqa: E402

from test_loading import BRANCHES, FakeEvent, opened  # noqa: E402


def loading_log(gi):
    log = []
    gi.loadingStarted.connect(lambda message, steps, file: log.append(('start', message, steps, file)))
    gi.loadingProgress.connect(lambda message, step: log.append(('step', message, step)))
    gi.loadingFinished.connect(lambda ok, message: log.append(('done', ok, message)))
    gi.eventChanged.connect(lambda: log.append(('event', gi.event())))
    return log


def stepping_scan(files, progress=None):
    ''' scan_files' steps for one file '''
    progress('Reading a.root', 0)
    progress('Opening with gallery', 1)
    return opened(files[0])


def test_open_reports_its_steps(qtbot):
    gi = GalleryInterface(scan=stepping_scan)
    log = loading_log(gi)
    assert gi.set_input_files(['/data/a.root'])
    assert log == [('start', 'Opening a.root', 3, True),
                   ('step', 'Reading a.root', 0), ('step', 'Opening with gallery', 1),
                   ('step', 'Drawing the first event', 2),
                   # the first event is drawn as part of the open
                   ('event', 1),
                   ('done', True, 'Opened a.root')]
    assert not gi.loading


def test_failed_open_reports_why(qtbot, capsys):
    def fail(files, progress=None):
        raise OpenError('could not open bad.root: corrupt')

    gi = GalleryInterface(scan=fail)
    log = loading_log(gi)
    assert not gi.set_input_files(['bad.root'])
    assert log[-1] == ('done', False, 'could not open bad.root: corrupt') and not gi.loading


def test_event_change_reports_loading(qtbot):
    gi = GalleryInterface(scan=lambda files, progress=None: opened(files[0]))
    gi.set_input_files(['a.root'])
    log = loading_log(gi)
    gi.next()
    assert log == [('start', 'Loading event 2 of 2', 0, False), ('event', 2), ('done', True, '')]


def test_no_open_or_navigation_while_loading(qtbot, capsys):
    ''' the GUI repaints between loading steps: a timer then can't move the event or open another file '''
    gi = GalleryInterface(scan=lambda files, progress=None: opened(files[0]))
    gi.set_input_files(['a.root'])
    attempts = []

    def meanwhile():
        attempts.append(gi.set_input_files(['b.root']))
        gi.next()
        attempts.append(gi.event())

    gi.eventChanged.connect(meanwhile)
    gi.next()
    # refused, and still the event being drawn
    assert attempts == [False, 2] and gi.current_file == 'a.root'
    assert 'still loading' in capsys.readouterr().out


@pytest.fixture
def gui(qtbot, monkeypatch, tmp_path):
    ''' a Gui with its settings in a temporary file, not the user's '''
    settings = QtCore.QSettings
    path = str(tmp_path / 'titus.ini')
    monkeypatch.setattr(QtCore, 'QSettings', lambda *args: settings(path, settings.IniFormat))
    from titus.gui import Gui
    gi = GalleryInterface()
    window = Gui(gi)
    qtbot.addWidget(window)
    return window


def test_overlay_and_bar_while_a_file_opens(gui):
    gi = gui.gallery_interface
    seen = []

    def scan(files, progress=None):
        progress('Reading a.root', 0)
        seen.append((gui._loading_overlay.isVisible(), gui._loading_bar.isVisible(),
                     gui._loading_bar.value(), gui._loading_bar.maximum(), gui.statusBar().currentMessage()))
        return opened(files[0])

    gi._scan = scan
    assert gi.set_input_files(['a.root'])
    assert seen == [(True, True, 0, 3, 'Reading a.root...')]
    # the overlay leaves the status bar, with the progress, uncovered
    assert gui._loading_overlay.geometry().bottom() < gui.statusBar().geometry().top()
    assert not gui._loading_overlay.isVisible() and not gui._loading_bar.isVisible()
    assert gui.statusBar().currentMessage() == 'Opened a.root'
    assert QtWidgets.QApplication.overrideCursor() is None


def test_failed_open_message_stays(gui):
    def fail(files, progress=None):
        raise OpenError('bad.root is not a .root file')

    gui.gallery_interface._scan = fail
    gui.gallery_interface.set_input_files(['bad.root'])
    assert gui.statusBar().currentMessage() == 'Could not open: bad.root is not a .root file'
    assert not gui._loading_overlay.isVisible()


def busy_cursor():
    cursor = QtWidgets.QApplication.overrideCursor()
    return cursor is not None and cursor.shape() == QtCore.Qt.WaitCursor


def test_event_load_shows_the_bar_and_cursor_only(gui):
    gi = gui.gallery_interface
    gi._scan = lambda files, progress=None: opened(files[0])
    gi.set_input_files(['a.root'])
    seen = []
    gi.eventChanged.connect(lambda: seen.append((gui._loading_overlay.isVisible(), gui._loading_bar.isVisible(),
                                                 gui._loading_bar.maximum(), busy_cursor())))
    gi.next()
    # a busy bar (no steps) and the wait cursor, no overlay
    assert seen == [(False, True, 0, True)] and not gui._loading_bar.isVisible()
    assert QtWidgets.QApplication.overrideCursor() is None


def products_of(type_name):
    return product_table(BRANCHES)['all'].get(type_name)


def test_boxes_without_products_disabled(qtbot):
    owner = mock.Mock()
    box = recoBox(owner, 'Hit', 'recob::Hit', None)
    assert not box.isEnabled() and 'No recob::Hit' in box.toolTip()
    box.set_product_and_producers('recob::Hit', products_of('recob::Hit'))
    assert box.isEnabled() and box.toolTip() == ''
    box.selectStage('Reco2')
    assert not box.isEnabled()
    box.selectStage('Reco1')
    assert box.isEnabled()

    multi = MultiSelectionBox(owner, 'recob::Hit', None)
    assert not multi.isEnabled()
    multi.set_products(products_of('recob::Hit'))
    assert multi.isEnabled() and multi.selected_products() == ['gaushit::Reco1']
    multi.set_products(())
    assert not multi.isEnabled() and multi.selected_products() == []


def wire_module(qtbot, available, detector='sbnd'):
    ''' a TpcModule with only its wire controls; available: the kinds the file has '''
    from titus.modules.tpc_module import TpcModule
    module = TpcModule.__new__(TpcModule)
    module._gm = mock.Mock()
    module._gm.current_geom.name.return_value = detector
    module._wires_off_by_user = False
    # kept: the buttons go with it
    parent = module._test_parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    layout = QtWidgets.QVBoxLayout(parent)
    for kind in ('none_wire', 'wire', 'raw_digit', 'channel_roi'):
        button = QtWidgets.QRadioButton(kind, parent)
        layout.addWidget(button)
        setattr(module, f'_{kind}_button', button)
    for kind, name in (('wire', 'recob::Hit'), ('raw_digit', 'raw::RawDigit'), ('channel_roi', 'recob::ChannelROI')):
        box = MultiSelectionBox(module, name, products_of('recob::Hit') if kind in available else None)
        setattr(module, f'_{kind}_choice', box)
    module._none_wire_button.setChecked(True)
    module._update_wire_buttons()
    return module


def checked(module):
    return [k for k in ('none_wire', 'wire', 'raw_digit', 'channel_roi') if getattr(module, f'_{k}_button').isChecked()]


@pytest.mark.parametrize('detector, available, expected', [
    ('sbnd', {'wire', 'raw_digit'}, 'wire'),
    ('sbnd', {'raw_digit'}, 'raw_digit'),
    ('icarus', {'channel_roi', 'raw_digit'}, 'channel_roi'),
    ('icarus', {'wire', 'channel_roi'}, 'wire'),
    # SBND has no channel ROI button
    ('sbnd', {'channel_roi'}, 'none_wire'),
    ('sbnd', set(), 'none_wire'),
])
def test_new_file_draws_its_wires(qtbot, detector, available, expected):
    module = wire_module(qtbot, available, detector)
    module._choose_wire_view()
    assert checked(module) == [expected]


def test_wire_buttons_follow_their_boxes(qtbot):
    module = wire_module(qtbot, {'raw_digit'})
    assert not module._wire_button.isEnabled() and module._raw_digit_button.isEnabled()
    assert 'No recob::Hit' in module._wire_button.toolTip()


def test_kind_chosen_or_none_by_the_user_kept(qtbot):
    module = wire_module(qtbot, {'wire', 'raw_digit'})
    module._raw_digit_button.setChecked(True)
    module._choose_wire_view()
    assert checked(module) == ['raw_digit']

    module._none_wire_button.setChecked(True)
    module._wires_off_by_user = True
    module._choose_wire_view()
    assert checked(module) == ['none_wire']


class DatedEvent(FakeEvent):
    ''' FakeEvent of data or simulation, with a time stamp (2025-03-01 12:00 UTC) '''
    def __init__(self, files, real_data):
        super().__init__(files)
        self.real_data = real_data

    def eventAuxiliary(self):
        aux = super().eventAuxiliary()
        aux.isRealData = lambda: self.real_data
        aux.time = lambda: mock.Mock(value=lambda: 1740830400 << 32)
        return aux


def dated_file(name, real_data, events):
    return OpenedFiles(files=[name], event=DatedEvent([(name, events)], real_data),
                       run_list=[{'run': r, 'subrun': s, 'event': e} for r, s, e in events],
                       key_table=product_table(BRANCHES))


def test_views_follow_the_event_and_file(gui):
    ''' the logo label and the wire waveforms: of the current event, after next and a new file '''
    import numpy as np
    from titus.modules import GeometryModule, LArSoftModule, TpcModule, ViewSelectModule
    gi = gui.gallery_interface
    lsm = LArSoftModule('sbnd', None)
    gm = GeometryModule(lsm)
    gm.add_gallery_interface(gi)
    views = ViewSelectModule()
    gui.add_module(views)
    tpc = TpcModule(lsm, gm)
    views.add_module(tpc, view_name='TPC')
    gui.add_module(lsm)
    gui.show()
    QtWidgets.QApplication.processEvents()
    tpc._logoOption.setChecked(True)
    view = tpc._wire_views[(2, 0)]

    def label():
        return view._label.toPlainText().split('\n')

    files = {'mc.root': dated_file('mc.root', False, [(1, 7, 100), (1, 7, 101)]),
             'data.root': dated_file('data.root', True, [(18255, 3, 42)])}
    gi._scan = lambda names, progress=None: files[names[0]]
    gi.set_input_files(['mc.root'])
    assert label() == ['SBND Simulation', 'Run 1, Subrun 7, Event 100']
    # a waveform clicked, but no wire image in the next event
    tpc._drawWireOption.setChecked(True)
    tpc.drawWireOnPlot(np.zeros(10), wire=400, plane=2, tpc=0, cryo=0, drawer=view)
    assert list(tpc._wirePlotItems) == [(0, 400)]
    gi.next()
    assert label() == ['SBND Simulation', 'Run 1, Subrun 7, Event 101']
    assert tpc._wirePlotItems == {} and view._wire_marks == []
    gi.set_input_files(['data.root'])
    assert label() == ['SBND Data', 'Run 18255, Subrun 3, Event 42', 'March 01, 2025']


def tpc_gui(gui, gdml=None):
    ''' (gallery interface, geometry module, TPC module) of a GUI drawing SBND '''
    from titus.modules import GeometryModule, LArSoftModule, TpcModule, ViewSelectModule
    gi = gui.gallery_interface
    lsm = LArSoftModule('sbnd', None)
    gm = GeometryModule(lsm, gdml=gdml)
    gm.add_gallery_interface(gi)
    views = ViewSelectModule()
    gui.add_module(views)
    tpc = TpcModule(lsm, gm)
    views.add_module(tpc, view_name='TPC')
    gui.add_module(lsm)
    gui.show()
    QtWidgets.QApplication.processEvents()
    return gi, gm, tpc


def made_with(name, gdml):
    ''' a simulated file made with this GDML (None: one whose configuration can't be read) '''
    from dataclasses import replace
    return replace(dated_file(name, False, [(1, 7, 100)]), gdml=gdml)


def test_files_are_drawn_with_the_geometry_they_were_made_with(gui):
    gi, gm, tpc = tpc_gui(gui)
    geom = gm.current_geom
    assert geom.gdml() == 'sbnd_v02_04'
    switched = []
    gm.detectorChanged.connect(lambda: switched.append(geom.gdml()))
    # the geometry, when the file's first event is drawn
    drawn = []
    gi.eventChanged.connect(lambda: drawn.append(tpc._wire_views[(2, 0)]._geometry.gdml()))
    files = {name: made_with(name, gdml) for name, gdml in
             [('v14.root', 'sbnd_v02_06_nowires'), ('v06.root', 'sbnd_v02_04'), ('remote.root', None),
              ('future.root', 'sbnd_v03_00')]}
    gi._scan = lambda names, progress=None: files[names[0]]
    gi.set_input_files(['v14.root'])
    # in place: the views drew the file with the GDML it was made with
    assert gm.current_geom is geom and geom.gdml() == 'sbnd_v02_06'
    assert switched == ['sbnd_v02_06'] and drawn == ['sbnd_v02_06']
    gi.set_input_files(['v06.root'])
    assert switched == ['sbnd_v02_06', 'sbnd_v02_04'] and drawn[-1] == 'sbnd_v02_04'
    # unknown: the geometry stays
    gi.set_input_files(['v14.root'])
    for name in ('remote.root', 'future.root'):
        gi.set_input_files([name])
        assert geom.gdml() == 'sbnd_v02_06'
    assert len(switched) == 3


def test_crt_outlines_follow_the_geometry(qtbot):
    from titus.geometry import Geometry, load_detector
    from titus.modules.crt_module import CrtViewWidget
    def outlines(view):
        return [tuple(r.rect().getRect()) for r in view._drawn_crt_modules]

    geom = Geometry.load('sbnd')
    view = CrtViewWidget(geom)
    qtbot.addWidget(view)
    before = outlines(view)
    geom.set_detector(load_detector('sbnd', 'sbnd_v02_06'))
    view.refresh_geometry()
    fresh = CrtViewWidget(Geometry.load('sbnd', 'sbnd_v02_06'))
    qtbot.addWidget(fresh)
    # as drawn for v02_06 from the start (the modules 4.2 cm lower in z), and drawn once
    assert outlines(view) == outlines(fresh) != before
    assert sum(isinstance(i, QtWidgets.QGraphicsRectItem) for i in view._plot.items) == len(before)


def test_a_geometry_given_on_the_command_line_is_kept(gui):
    gi, gm, tpc = tpc_gui(gui, gdml='sbnd_v02_06')
    assert gm.current_geom.gdml() == 'sbnd_v02_06'
    gi._scan = lambda names, progress=None: made_with(names[0], 'sbnd_v02_04')
    gi.set_input_files(['v06.root'])
    assert gm.current_geom.gdml() == 'sbnd_v02_06'
