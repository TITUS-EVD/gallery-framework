"""
GalleryReader's python side: event indexing across files, navigation,
product discovery and column conversion, on a fake ROOT whose titusio
functions fill the column structs as UserDev/EventDisplay/Reader does.
The compiled library itself is checked on a gpvm with
tools/check_gallery_reader.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from titus.readers import EventID, GalleryReader, Hits, ProductNotFound, RawDigits, WireSignals
from titus.readers.gallery import EXTRACTORS, columns_to_container


def hit_columns(n):
    return dict(channel=list(range(n)), cryostat=[0] * n, tpc=[1] * n, plane=[2] * n,
                wire=list(range(n)), peak_time=[100.5] * n, rms=[2.] * n, integral=[50.] * n,
                peak_amplitude=[10.] * n, start_tick=[95] * n, end_tick=[106] * n)


WIRE_COLUMNS = dict(channel=[3, 4], n_ticks=[50, 50], roi_row=[0, 0, 1], roi_start=[1, 20, 5],
                    roi_offset=[0, 2, 3, 5], values=[1., 2., 3., 4., 5.])
RAW_COLUMNS = dict(channel=[7, 8], pedestal=[400., 2048.], n_samples=[3, 2], adc=[1, 2, 3, 4, 5, 0], width=3)

FILES = {
    'a.root': {
        'branches': ['EventAuxiliary', 'recob::Hits_gaushit__Reco1.', 'recob::Wires_simtpc2d_dnnsp_DetSim.',
                     'recob::Hitrecob::Wirevoidart::Assns_gaushit__Reco1.'],
        'events': [((1, 1, 1), {('hits', 'gaushit::Reco1'): hit_columns(3),
                                ('wires', 'simtpc2d:dnnsp:DetSim'): WIRE_COLUMNS}),
                   ((1, 1, 2), {('hits', 'gaushit::Reco1'): hit_columns(5)})],
    },
    'b.root': {
        'branches': ['recob::Hits_gaushit__Reco1.', 'raw::RawDigits_daq__DetSim.'],
        'events': [((2, 5, 9), {('hits', 'gaushit::Reco1'): hit_columns(1),
                                ('raw_digits', 'daq::DetSim'): RAW_COLUMNS})],
    },
}


class FakeEvent:
    """ gallery.Event over one file of FILES """
    log = []

    def __init__(self, files, has_go_to_entry=True):
        (self.path,) = files
        self.file = FILES[self.path]
        self.entry = 0
        if has_go_to_entry:
            self.goToEntry = self._go_to_entry

    def atEnd(self):
        return self.entry >= len(self.file['events'])

    def next(self):
        self.entry += 1

    def toBegin(self):
        self.log.append(('toBegin', self.path))
        self.entry = 0

    def _go_to_entry(self, entry):
        self.log.append(('goToEntry', self.path, entry))
        self.entry = entry

    def eventEntry(self):
        return self.entry

    def eventAuxiliary(self):
        run, subrun, event = self.file['events'][self.entry][0]
        return SimpleNamespace(run=lambda: run, subRun=lambda: subrun, event=lambda: event)

    def getTFile(self):
        branches = [SimpleNamespace(GetName=lambda b=b: b) for b in self.file['branches']]
        return SimpleNamespace(Get=lambda name: SimpleNamespace(GetListOfBranches=lambda: branches))

    def products(self):
        return self.file['events'][self.entry][1]


def make_root(has_go_to_entry=True):
    def reader(kind):
        def read(ev, tag, out):
            data = ev.products().get((kind, tag))
            if data is None:
                return False
            for name, values in data.items():
                setattr(out, name, list(values) if isinstance(values, list) else values)
            return True
        return read

    lib = SimpleNamespace(**{function: reader(kind) for kind, (function, _) in EXTRACTORS.items()},
                          **{struct: SimpleNamespace for _, struct in EXTRACTORS.values()})
    return SimpleNamespace(
        titusio=lib,
        gallery=SimpleNamespace(Event=lambda files: FakeEvent(files, has_go_to_entry)),
        std=SimpleNamespace(vector={'string': list}),
    )


@pytest.fixture
def reader():
    FakeEvent.log.clear()
    return GalleryReader(['a.root', 'b.root'], root=make_root())


def test_events_across_files(reader):
    assert reader.events == [EventID(1, 1, 1), EventID(1, 1, 2), EventID(2, 5, 9)]
    assert reader.entry_of(2, 5, 9) == 2
    with pytest.raises(LookupError):
        reader.entry_of(9, 9, 9)


def test_tags_from_branches(reader):
    assert reader.tags('hits') == ['gaushit::Reco1']
    assert reader.tags('wires') == ['simtpc2d:dnnsp:DetSim']
    assert reader.tags('raw_digits') == ['daq::DetSim']
    assert reader.tags('raw_digits', entry=0) == []
    assert reader.tags('wires', entry=2) == []


def test_read_products(reader):
    hits = reader.hits(0, 'gaushit')
    assert isinstance(hits, Hits) and len(hits) == 3 and hits.tpc.dtype == np.int32
    assert len(reader.hits(1, 'gaushit')) == 5
    assert len(reader.hits(2, 'gaushit')) == 1
    wires = reader.wires(0, 'simtpc2d:dnnsp')
    assert wires.dense(0)[[1, 2, 20]].tolist() == [1, 2, 3] and wires.dense(1)[5:7].tolist() == [4, 5]
    digits = reader.raw_digits(2, 'daq')
    assert digits.adc.tolist() == [[1, 2, 3], [4, 5, 0]] and digits.n_samples.tolist() == [3, 2]


def test_product_missing_in_event(reader):
    """ listed for the file (branch exists) but not stored for this event """
    with pytest.raises(ProductNotFound, match='not in event'):
        reader.wires(1, 'simtpc2d:dnnsp')


def test_navigation(reader):
    reader.hits(2, 'gaushit')
    reader.hits(0, 'gaushit')
    reader.hits(0, 'gaushit')
    # indexing leaves each file at its end; then one jump per change of entry
    assert FakeEvent.log == [('goToEntry', 'b.root', 0), ('goToEntry', 'a.root', 0)]


def test_navigation_without_go_to_entry():
    FakeEvent.log.clear()
    reader = GalleryReader(['a.root'], root=make_root(has_go_to_entry=False))
    assert len(reader.hits(1, 'gaushit')) == 5
    assert len(reader.hits(0, 'gaushit')) == 3
    assert FakeEvent.log == [('toBegin', 'a.root'), ('toBegin', 'a.root')]


def test_empty_columns():
    empty = SimpleNamespace(channel=[], n_ticks=[], roi_row=[], roi_start=[], roi_offset=[], values=[])
    assert len(columns_to_container('wires', empty)) == 0
    raw = SimpleNamespace(channel=[], pedestal=[], n_samples=[], adc=[], width=0)
    assert columns_to_container('raw_digits', raw).adc.shape == (0, 0)
