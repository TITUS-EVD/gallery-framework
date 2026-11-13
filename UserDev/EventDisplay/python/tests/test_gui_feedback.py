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


def made_with(name, gdml, sce_map=None):
    ''' a simulated file made with this GDML and space charge map (None: one whose configuration can't be read) '''
    from dataclasses import replace
    return replace(dated_file(name, False, [(1, 7, 100)]), gdml=gdml, sce_map=sce_map)


def test_space_charge_follows_the_file(gui, capsys):
    gi, gm, tpc = tpc_gui(gui)
    sbnd_map = 'SCEoffsets_SBND_E500_dualmap_CV_voxelTH3'
    files = {'sce.root': made_with('sce.root', 'sbnd_v02_06', sbnd_map),
             'sce2.root': made_with('sce2.root', 'sbnd_v02_06', sbnd_map),
             'nosce.root': made_with('nosce.root', 'sbnd_v02_06'),
             'other.root': made_with('other.root', 'sbnd_v02_06', 'SCEoffsets_ICARUS_unknown')}
    gi._scan = lambda names, progress=None: files[names[0]]
    option = tpc._space_charge
    # before a file: nothing to apply
    assert tpc.space_charge_map() is None
    gi.set_input_files(['sce.root'])
    assert option.isEnabled() and option.isChecked() and tpc.space_charge_map().name == sbnd_map
    assert sbnd_map in option.toolTip()
    # turned off: kept for the next file
    option.setChecked(False)
    assert tpc.space_charge_map() is None
    gi.set_input_files(['sce2.root'])
    assert option.isEnabled() and not option.isChecked() and tpc.space_charge_map() is None
    option.setChecked(True)
    # no space charge in the simulation, or a map TITUS doesn't have: as stored
    gi.set_input_files(['nosce.root'])
    assert not option.isEnabled() and tpc.space_charge_map() is None
    capsys.readouterr()
    gi.set_input_files(['other.root'])
    assert not option.isEnabled() and tpc.space_charge_map() is None
    assert 'SCEoffsets_ICARUS_unknown' in capsys.readouterr().out and 'dump_sce_map.py' in option.toolTip()


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


def test_mc_annotations_need_no_mctrack_box(gui):
    ''' the MC truth dock annotates with the file's MCTracks, or those chosen in the MCTrack box '''
    from dataclasses import replace
    gi, gm, tpc = tpc_gui(gui)
    mctracks = [('sim::MCTracks_mcreco__G4.', 'vector<sim::MCTrack>'),
                ('sim::MCTracks_other__G4.', 'vector<sim::MCTrack>')]
    files = {'mc.root': replace(dated_file('mc.root', False, [(1, 7, 100)]), key_table=product_table(BRANCHES + mctracks)),
             'reco.root': dated_file('reco.root', False, [(1, 7, 100)])}
    gi._scan = lambda names, progress=None: files[names[0]]
    gi.set_input_files(['mc.root'])
    annotations = tpc._mc_annotations
    assert annotations in tpc._drawables and annotations not in tpc._product_box_map.values()
    # no MCTrack chosen in the box: all of them
    box = next(b for b, product in tpc._box_products.items() if product == 'sim::MCTrack')
    assert tpc._product_box_map[box].producers() == []
    assert sorted(annotations.producers()) == ['mcreco::G4', 'other::G4']
    # one chosen: that one, as drawn
    box.setCurrentIndex(box.findText('mcreco'))
    box.activated[str].emit('mcreco')
    assert annotations.producers() == ['mcreco::G4']
    box.activated[str].emit('--Select--')
    assert sorted(annotations.producers()) == ['mcreco::G4', 'other::G4']
    # a file without: nothing to annotate with
    gi.set_input_files(['reco.root'])
    assert annotations.producers() == [] and tpc.annotate_mc_particles([]) is None


def test_annotation_font_size_setting(gui):
    gi, gm, tpc = tpc_gui(gui)
    assert tpc.annotation_font_size() == 12
    redrawn = []
    tpc._mc_annotations.drawObjects = lambda: redrawn.append(tpc.annotation_font_size())
    tpc._annotation_font_size.setValue(20)
    assert redrawn == [20] and int(tpc._settings.value('TPC/MC annotation font size')) == 20
    tpc.restore_settings_defaults()
    assert tpc.annotation_font_size() == 12 and redrawn[-1] == 12
    # labels along their paths by default; level kept, and redrawn
    assert tpc.annotation_labels_rotated()
    count = len(redrawn)
    tpc._annotation_rotate.setChecked(False)
    assert not tpc.annotation_labels_rotated() and len(redrawn) == count + 1
    assert tpc._settings.value('TPC/Rotate MC annotation labels', type=bool) is False
    tpc.restore_from_settings()
    assert not tpc.annotation_labels_rotated()
    tpc.restore_settings_defaults()
    assert tpc.annotation_labels_rotated()
    # the annotation settings together, in their group box
    box = tpc._annotation_rotate.parentWidget()
    assert isinstance(box, QtWidgets.QGroupBox) and box is tpc._annotation_font_size.parentWidget()
    # which particles are drawn with their paths, as the dock says (None: all)
    tpc.annotate_mc_particles([])
    assert tpc.annotated_mc_paths() is None
    tpc.annotate_mc_particles([], paths=[])
    assert tpc.annotated_mc_paths() == []


def test_module_docks_where_they_were_last_session(gui, qtbot):
    ''' the window state is restored before the modules make their docks: they are put back where they were '''
    from titus.gui import Gui
    gi, gm, tpc = tpc_gui(gui)
    Qt = QtCore.Qt
    # by default: the MC truth dock on the right, the view controls on the left
    assert gui.dockWidgetArea(tpc._mctruth_dock) == Qt.RightDockWidgetArea
    gui.addDockWidget(Qt.LeftDockWidgetArea, tpc._mctruth_dock)
    gui.addDockWidget(Qt.BottomDockWidgetArea, tpc._view_dock)
    # closed while another view was shown (its docks hidden), as closeEvent saves it without quitting the tests
    tpc.deactivate()
    gui._settings.beginGroup('MainWindow')
    gui._settings.setValue('state', gui.saveState(1))
    gui._settings.endGroup()

    # next session: the same settings file (the fixture's)
    again = Gui(GalleryInterface())
    qtbot.addWidget(again)
    _, _, tpc2 = tpc_gui(again)
    assert again.dockWidgetArea(tpc2._mctruth_dock) == Qt.LeftDockWidgetArea
    assert again.dockWidgetArea(tpc2._view_dock) == Qt.BottomDockWidgetArea
    # the TPC view's docks shown with it; the MC truth dock hidden until there is MC truth, as before
    assert not tpc2._view_dock.isHidden() and tpc2._mctruth_dock.isHidden()


def _view_pixels(image, view, widget, scale):
    ''' the RGBA pixels (h, w, 4) of an image of widget (scale times its size) where view's viewport is '''
    import numpy as np
    image = image.toImage().convertToFormat(image.toImage().Format_RGBA8888)
    pixels = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), dtype=np.uint8)
    pixels = pixels.reshape(image.height(), image.bytesPerLine() // 4, 4)[:, :image.width()]
    corner = view.viewport().mapTo(widget, QtCore.QPoint(0, 0))
    size = view.viewport().size()
    x0, y0 = round(corner.x() * scale), round(corner.y() * scale)
    return pixels[y0:y0 + round(size.height() * scale), x0:x0 + round(size.width() * scale)]


def test_annotation_layer_in_screenshots(gui, monkeypatch, tmp_path):
    ''' with the setting, a view screenshot saved to a file puts the annotations in a transparent image of their own '''
    from PyQt5 import QtGui
    from titus.drawables.annotation import annotation_items
    from titus.drawables.drawable import add_items
    gi, gm, tpc = tpc_gui(gui)
    monkeypatch.chdir(tmp_path)
    view = next(v for v in tpc._wire_views.values() if v.isVisible())
    # a black view, so the white annotations would show in it
    view.setBackground('k')
    view._view.disableAutoRange()
    view._view.setRange(QtCore.QRectF(0, 0, 100, 100), padding=0)
    items = annotation_items([[(10, 50), (90, 50)]], 'mu-')
    add_items(view._view, items)
    tpc._mc_annotations._annotations = list(items)
    QtWidgets.QApplication.processEvents()
    widget, scale = gui._central_widget, 2.0
    gui._settings.setValue('General/Screenshot scale', scale)
    gui._settings.setValue('General/Screenshot mode', 'File')
    captured = []
    monkeypatch.setattr(QtGui.QPixmap, 'save', lambda self, file, fmt: captured.append((file.fileName(), QtGui.QPixmap(self))) or True)

    # off by default: one image, the annotations in it
    assert not tpc._annotation_layer.isChecked() and tpc.screenshot_layer() is None
    gui._on_capture_action()
    assert len(captured) == 1
    white = (_view_pixels(captured[0][1], view, widget, scale)[..., :3] > 200).all(axis=-1)
    assert white.any()

    tpc._annotation_layer.setChecked(True)
    assert tpc._settings.value('TPC/Separate MC annotation layer in screenshots', type=bool) is True
    assert tpc.screenshot_layer() == ('annotations', items)
    captured.clear()
    gui._on_capture_action()
    (name, base), (layer_name, layer) = captured
    assert layer_name == name[:-len('.png')] + '_annotations.png'
    assert layer.size() == base.size() and layer.hasAlphaChannel()
    # the wire planes without the annotations
    assert not (_view_pixels(base, view, widget, scale)[..., :3] > 200).all(axis=-1).any()
    # the annotations alone, where they were drawn, on nothing
    pixels = _view_pixels(layer, view, widget, scale)
    drawn = pixels[..., 3] > 0
    assert (drawn == white).mean() > 0.99
    image = layer.toImage()
    assert image.pixelColor(0, 0).alpha() == 0 and image.pixelColor(image.width() - 1, image.height() - 1).alpha() == 0
    # all as it was
    assert all(item.opacity() == 1 for item in items) and view.backgroundBrush().color() == QtGui.QColor(QtCore.Qt.black)
    assert all(item.opacity() == 1 for item in view._view.addedItems)

    # the window, and the clipboard: one image
    captured.clear()
    gui._on_capture_screen_action()
    assert len(captured) == 1
    gui._settings.setValue('General/Screenshot mode', 'Clipboard')
    gui._on_capture_action()
    assert len(captured) == 1
    clip = QtWidgets.QApplication.clipboard().pixmap()
    assert (_view_pixels(clip, view, widget, scale)[..., :3] > 200).all(axis=-1).any()

    # back to one image, kept between sessions
    tpc.restore_settings_defaults()
    assert not tpc._annotation_layer.isChecked()
    box = tpc._annotation_layer.parentWidget()
    assert isinstance(box, QtWidgets.QGroupBox) and box is tpc._annotation_rotate.parentWidget()


def test_logo_and_scale_bar_in_the_annotation_layer(gui, monkeypatch, tmp_path):
    ''' the separate annotation layer takes the views' logos, labels and scale bars too, even with no annotations '''
    from PyQt5 import QtGui
    gi, gm, tpc = tpc_gui(gui)
    monkeypatch.chdir(tmp_path)
    view = next(v for v in tpc._wire_views.values() if v.isVisible())
    tpc._annotation_layer.setChecked(True)
    # nothing over the wire planes: one image
    assert tpc.screenshot_layer() is None

    tpc._logoOption.setChecked(True)
    tpc._scaleBarOption.setChecked(True)
    QtWidgets.QApplication.processEvents()
    overlays = view.overlay_items()
    assert set(overlays) == {view._logo, view._label, view._xBar}
    name, items = tpc.screenshot_layer()
    assert name == 'annotations' and set(overlays) <= set(items)

    widget, scale = gui._central_widget, 1.0
    gui._settings.setValue('General/Screenshot scale', scale)
    gui._settings.setValue('General/Screenshot mode', 'File')
    captured = []
    monkeypatch.setattr(QtGui.QPixmap, 'save', lambda self, file, fmt: captured.append((file.fileName(), QtGui.QPixmap(self))) or True)
    gui._on_capture_action()
    (_, base), (layer_name, layer) = captured
    assert layer_name.endswith('_annotations.png')
    # the logo and scale bar drawn in the layer, at least where the view is
    assert (_view_pixels(layer, view, widget, scale)[..., 3] > 0).any()
    # and painted again as before, on screen
    assert all(item.opacity() == 1 for item in overlays)

    # hidden: not in the layer
    tpc._logoOption.setChecked(False)
    tpc._scaleBarOption.setChecked(False)
    assert view.overlay_items() == [] and tpc.screenshot_layer() is None


def test_secondary_paths_option_kept_between_sessions(gui, qtbot):
    ''' the MC truth dock's Secondaries' paths: on at first, re-annotates when toggled, remembered '''
    from titus.gui import Gui
    gi, gm, tpc = tpc_gui(gui)
    panel = tpc._mctruth_panel
    assert panel.secondary_paths.isChecked() and tpc.annotated_mc_secondary_paths() is True
    with qtbot.waitSignal(panel.particlesAnnotated):
        panel.secondary_paths.setChecked(False)
    assert tpc.annotated_mc_secondary_paths() is False
    assert tpc._settings.value('TPC/Draw MC annotated secondaries paths', type=bool) is False

    # next session: the same settings file (the fixture's)
    again = Gui(GalleryInterface())
    qtbot.addWidget(again)
    _, _, tpc2 = tpc_gui(again)
    assert not tpc2._mctruth_panel.secondary_paths.isChecked() and tpc2.annotated_mc_secondary_paths() is False


def test_neutrino_path_option_and_screenshot_layer(gui, qtbot, monkeypatch):
    ''' the MC truth dock's Neutrino path: off at first, remembered; its annotation in the separate layer '''
    from types import SimpleNamespace
    from titus.gui import Gui
    gi, gm, tpc = tpc_gui(gui)
    panel = tpc._mctruth_panel
    assert not panel.show_neutrino.isChecked()
    panel.show_neutrino.setChecked(True)
    assert tpc._settings.value('TPC/Draw MC neutrino path', type=bool) is True
    again = Gui(GalleryInterface())
    qtbot.addWidget(again)
    _, _, tpc2 = tpc_gui(again)
    assert tpc2._mctruth_panel.show_neutrino.isChecked()

    # the neutrino's line and name: with the annotations in their own image
    nu_items = [QtWidgets.QGraphicsLineItem(), QtWidgets.QGraphicsLineItem()]
    redrawn = []
    drawable = SimpleNamespace(drawn_items=lambda: nu_items, draw_neutrino=lambda: redrawn.append(1))
    monkeypatch.setattr(tpc, '_mctruth_drawables', lambda: [drawable])
    tpc._annotation_layer.setChecked(True)
    assert tpc.screenshot_layer() == ('annotations', nu_items)
    # drawn again with the other annotations, e.g. another font size
    tpc._refresh_mc_annotations()
    assert redrawn == [1]
