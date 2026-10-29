"""
Opening files: TFile then gallery (scan_files), in a background thread
(GalleryInterface.open_files) while the GUI keeps running. gallery is faked.
"""

import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

sys.modules.setdefault('ROOT', mock.MagicMock())
pytest.importorskip('PyQt5')
from PyQt5 import QtCore  # noqa: E402

from titus.gallery_interface.gallery_interface import (GalleryInterface, OpenedFiles, OpenError,  # noqa: E402
                                                       product_table, readable_files, scan_files)

BRANCHES = [
    ('EventAuxiliary', 'art::EventAuxiliary'),
    ('recob::Hits_gaushit__Reco1.', 'vector<recob::Hit>'),
    ('recob::Tracks_pandoraTrack__Reco2.', 'vector<recob::Track>'),
    ('recob::Hitrecob::Trackvoidart::Assns_pandoraTrack__Reco2.', 'art::Assns<recob::Hit,recob::Track,void>'),
]


class FakeTree:
    ''' an Events tree: its branches, and EventAuxiliary filled by GetEntry '''
    def __init__(self, events, branches):
        self._events, self._branches = events, branches
        self.EventAuxiliary = None
        tree = self

        class AuxBranch:
            def GetEntries(self):
                return len(tree._events)

            def GetEntry(self, i):
                run, subrun, event = tree._events[i]
                tree.EventAuxiliary = SimpleNamespace(run=lambda: run, subRun=lambda: subrun, event=lambda: event)

        self._aux = AuxBranch()

    def GetBranch(self, name):
        assert name == 'EventAuxiliary'
        return self._aux

    def GetListOfBranches(self):
        return [SimpleNamespace(GetName=lambda n=n: n, GetTypeName=lambda t=t: t) for n, t in self._branches]


class FakeTFile:
    def __init__(self, name, events, branches=BRANCHES, log=None):
        self._name, self._tree, self._log = name, FakeTree(events, branches), log

    def IsZombie(self):
        return False

    def Get(self, name):
        assert name == 'Events'
        return self._tree

    def Close(self):
        if self._log is not None:
            self._log.append(('close', self._name))


class FakeVector(list):
    def push_back(self, value):
        self.append(value)


class FakeEvent:
    ''' a gallery.Event over files of (name, [(run, subrun, event)]) '''
    def __init__(self, files, branches=BRANCHES):
        self._entries = [(name, ids) for name, events in files for ids in events]
        self._files = {name: FakeTFile(name, branches) for name, _ in files}
        self.entry = 0

    def atEnd(self):
        return self.entry >= len(self._entries)

    def next(self):
        self.entry += 1

    def toBegin(self):
        self.entry = 0

    def goToEntry(self, entry):
        self.entry = entry

    def eventEntry(self):
        return self.entry

    def getTFile(self):
        return self._files[self._entries[self.entry][0]]

    def eventAuxiliary(self):
        run, subrun, event = self._entries[self.entry][1]
        return SimpleNamespace(run=lambda: run, subRun=lambda: subrun, event=lambda: event)


def fake_root(files, log=None):
    ''' ROOT whose TFile and gallery.Event open these (name, events) files; log records the order '''
    by_name = dict(files)
    log = [] if log is None else log

    def open_tfile(name):
        log.append(('TFile', name))
        return FakeTFile(name, by_name[name], log=log)

    def open_event(names):
        log.append(('gallery', list(names)))
        return FakeEvent([(n, by_name[n]) for n in names])

    return SimpleNamespace(std=SimpleNamespace(vector={'string': FakeVector}),
                           TFile=SimpleNamespace(Open=open_tfile),
                           gallery=SimpleNamespace(Event=open_event))


def test_product_table():
    table = product_table(BRANCHES + [('sbn::NuMuSelectionsbn::NuMuThingart::Assns_numu__Reco2.',
                                       'art::Assns<sbn::NuMuSelection,sbn::NuMuThing,void>'),
                                      ('sbn::NuMurecob::PFParticleart::Assns_numu__Reco2.',
                                       'art::Assns<sbn::NuMu,recob::PFParticle,void>')])
    assert sorted(table) == ['Reco1', 'Reco2', 'all']
    assert [p.full_name() for p in table['all']['recob::Hit']] == ['gaushit::Reco1']
    assert [p.full_name() for p in table['Reco2']['recob::Track']] == ['pandoraTrack::Reco2']
    # associations aren't listed, except NuMu selection ones not to PFParticles
    names = [p.name() for products in table['all'].values() for p in products]
    assert not any('recob::Hitrecob::Track' in n for n in names)
    assert any('NuMuThing' in n for n in names) and not any('PFParticle' in n for n in names)


def test_readable_files(tmp_path):
    good = tmp_path / 'a.root'
    good.write_text('')
    (tmp_path / 'b.txt').write_text('')
    keep, warnings = readable_files([str(good), str(tmp_path / 'b.txt'), str(tmp_path / 'missing.root'),
                                     'root://host//x.root'])
    assert keep == [str(good), 'root://host//x.root']
    assert len(warnings) == 2 and 'not a .root file' in warnings[0] and 'does not exist' in warnings[1]


def test_scan_files_reads_with_tfile_before_gallery(tmp_path):
    ''' gallery's constructor crashed in the GUI when it opened a file first: TFile reads each file before '''
    names = [str(tmp_path / f) for f in ('a.root', 'b.root')]
    for n in names:
        open(n, 'w').close()
    log = []
    root = fake_root([(names[0], [(1, 1, 1), (1, 1, 2)]), (names[1], [(1, 2, 7)])], log)
    result = scan_files(names, root)
    assert log == [('TFile', names[0]), ('close', names[0]), ('TFile', names[1]), ('close', names[1]),
                   ('gallery', names)]
    assert result.run_list == [{'run': 1, 'subrun': 1, 'event': 1}, {'run': 1, 'subrun': 1, 'event': 2},
                               {'run': 1, 'subrun': 2, 'event': 7}]
    # products of both files, and the event at the start
    assert len(result.key_table['all']['recob::Hit']) == 2
    assert result.event.eventEntry() == 0 and result.files == names


def test_scan_files_errors(tmp_path):
    empty = str(tmp_path / 'empty.root')
    open(empty, 'w').close()
    with pytest.raises(OpenError, match='no events'):
        scan_files([empty], fake_root([(empty, [])]))
    with pytest.raises(OpenError, match='does not exist'):
        scan_files([str(tmp_path / 'missing.root')], fake_root([]))

    def broken(names):
        raise RuntimeError('xrootd says no')

    root = fake_root([(empty, [(1, 1, 1)])])
    root.gallery.Event = broken
    with pytest.raises(OpenError, match='xrootd says no'):
        scan_files([empty], root)


def opened(name, events=((1, 1, 1), (1, 1, 2))):
    return OpenedFiles(files=[name], event=FakeEvent([(name, list(events))]),
                       run_list=[{'run': r, 'subrun': s, 'event': e} for r, s, e in events],
                       key_table=product_table(BRANCHES))


def recorder(gi):
    ''' gallery interface signals, in order '''
    log = []
    gi.fileChanged.connect(lambda: log.append(('file', gi.current_file)))
    gi.eventChanged.connect(lambda: log.append(('event', gi.event())))
    gi.loadingStarted.connect(lambda m: log.append(('started', m)))
    gi.loadingFinished.connect(lambda ok, m: log.append(('finished', ok)))
    return log


def test_open_without_threads(qtbot):
    gi = GalleryInterface(threaded=False, scan=lambda files: opened(files[0]))
    log = recorder(gi)
    assert gi.open_files(['a.root'])
    # file, then its first event, all before open_files returns
    assert log == [('file', 'a.root'), ('event', 1), ('finished', True)]
    assert gi.available_events() == [1, 2] and gi.get_products('recob::Hit')[0].full_name() == 'gaushit::Reco1'


def test_open_in_the_background(qtbot):
    release = threading.Event()
    scanned_in = []

    def slow_scan(files):
        scanned_in.append(threading.current_thread() is threading.main_thread())
        release.wait(5)
        return opened(files[0])

    gi = GalleryInterface(threaded=True, scan=slow_scan)
    gi._scan = lambda files: opened(files[0], events=[(5, 5, 5)])
    gi.set_input_files(['old.root'])
    gi._scan = slow_scan
    log = recorder(gi)

    assert gi.open_files(['new.root'])
    assert gi.is_loading() and log == [('started', 'Opening new.root...')]
    # while it opens: the GUI's event loop runs, the old file is kept but not touched
    ticks = []
    QtCore.QTimer.singleShot(10, lambda: ticks.append(1))
    qtbot.waitUntil(lambda: ticks == [1], timeout=2000)
    assert gi.current_file == 'old.root' and gi.event_handle() is None and gi.event() == 0
    assert not gi.open_files(['other.root'])  # one at a time
    gi.next()
    gi.process_event(True)
    assert log == [('started', 'Opening new.root...')]

    with qtbot.waitSignal(gi.loadingFinished, timeout=5000) as finished:
        release.set()
    assert finished.args == [True, 'Opened new.root']
    assert scanned_in == [False]  # scanned in the worker thread
    assert log[1:] == [('file', 'new.root'), ('event', 1), ('finished', True)]
    assert not gi.is_loading() and gi.event_handle() is not None


def test_failed_background_open_keeps_the_file(qtbot):
    gi = GalleryInterface(threaded=True, scan=lambda files: opened(files[0]))
    gi.set_input_files(['good.root'])

    def fail(files):
        time.sleep(0.05)
        raise OpenError('could not open bad.root: corrupt')

    gi._scan = fail
    log = recorder(gi)
    with qtbot.waitSignal(gi.loadingFinished, timeout=5000) as finished:
        assert gi.open_files(['bad.root'])
    assert finished.args == [False, 'could not open bad.root: corrupt']
    assert log == [('started', 'Opening bad.root...'), ('finished', False)]
    assert gi.current_file == 'good.root' and gi.event() == 1


def test_inactive_module_catches_up_on_file_change(qtbot):
    from titus.modules.module import Module

    class Recording(Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def on_file_change(self):
            self.calls.append('file')

        def update(self):
            self.calls.append('update')

    module = Recording()
    module.deactivate = lambda: setattr(module, '_active', False)
    module.deactivate()
    module._on_file_change()
    module._on_event_change()
    assert module.calls == []
    module.activate()
    # the new file's products, then one redraw
    assert module.calls == ['file', 'update']
    module.activate()
    assert module.calls == ['file', 'update']


def test_open_on_the_gui_thread_if_pyroot_polls_events(qtbot, monkeypatch, capsys):
    ''' PyROOT's event polling thread uses ROOT concurrently: a background open would crash '''
    import titus.gallery_interface.gallery_interface as gi_module
    from titus.gallery_interface.gallery_interface import root_event_thread_running
    stop = threading.Event()

    def _process_root_events():  # the name of PyROOT's polling function (ROOT/_application.py)
        stop.wait(5)

    assert not root_event_thread_running()
    poller = threading.Thread(target=_process_root_events, daemon=True)
    poller.start()
    try:
        assert root_event_thread_running()
        monkeypatch.setattr(gi_module, 'scan_files', lambda files: opened(files[0]))
        gi = GalleryInterface(threaded=True)
        log = recorder(gi)
        assert gi.open_files(['a.root'])
        # opened before open_files returned: no worker thread
        assert log == [('file', 'a.root'), ('event', 1), ('finished', True)]
        assert "PyROOT's event thread is running" in capsys.readouterr().out
    finally:
        stop.set()
        poller.join()


def test_root_setup_turns_off_the_event_thread(monkeypatch):
    from titus.gallery_interface.gallery_interface import _root
    root = SimpleNamespace(PyConfig=SimpleNamespace(StartGUIThread=True))
    monkeypatch.setitem(sys.modules, 'ROOT', root)
    assert _root() is root and root.PyConfig.StartGUIThread is False
