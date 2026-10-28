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

TRACK_COLUMNS = dict(id=[1], point_offset=[0, 2], x=[0., 1.], y=[0., 1.], z=[0., 1.],
                     valid=['\x01', '\x00'])  # unsigned char, as cppyy may iterate it
ASSOCIATION_COLUMNS = dict(left=[0, 0], right=[0, 1], left_product=[0, 0], right_product=[1, 1],
                           meta_index=[0, 1], product_tags=['pandoraTrack::Reco2', 'gaushit::Reco1'])

FILES = {
    'a.root': {
        'branches': ['EventAuxiliary', 'recob::Hits_gaushit__Reco1.', 'recob::Wires_simtpc2d_dnnsp_DetSim.',
                     'recob::Hitrecob::Wirevoidart::Assns_gaushit__Reco1.'],
        'events': [((1, 1, 1), {('hits', 'gaushit::Reco1'): hit_columns(3),
                                ('wires', 'simtpc2d:dnnsp:DetSim'): WIRE_COLUMNS}),
                   ((1, 1, 2), {('hits', 'gaushit::Reco1'): hit_columns(5)})],
    },
    'c.root': {
        'branches': ['recob::Hits_gaushit__Reco1.', 'recob::Tracks_pandoraTrack__Reco2.',
                     'recob::Hitrecob::Trackrecob::TrackHitMetaart::Assns_pandoraTrack__Reco2.'],
        'events': [((3, 1, 1), {('hits', 'gaushit::Reco1'): hit_columns(2),
                                ('tracks', 'pandoraTrack::Reco2'): TRACK_COLUMNS,
                                ('track_hits', 'pandoraTrack::Reco2'): ASSOCIATION_COLUMNS})],
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
        """ as gallery: only within the current file, which is gone at the end """
        if self.atEnd():
            raise RuntimeError('std::exception')
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
    association_calls = []

    def reader(kind):
        def read(ev, tag, *args):
            out = args[-1]
            if len(args) > 1:
                association_calls.append((tag, args[0], list(args[1])))
            data = ev.products().get((kind, tag))
            if data is None:
                return False
            for name, values in data.items():
                setattr(out, name, list(values) if isinstance(values, list) else values)
            return True
        return read

    functions = {}
    for kind, (function, _) in EXTRACTORS.items():
        functions.setdefault(function, []).append(kind)

    def dispatch(kinds):
        # several kinds share readAssociation: serve the one this event has
        readers = [reader(k) for k in kinds]

        def read(ev, tag, *args):
            for kind, r in zip(kinds, readers):
                if (kind, tag) in ev.products():
                    return r(ev, tag, *args)
            return readers[0](ev, tag, *args)
        return read

    lib = SimpleNamespace(**{function: dispatch(kinds) for function, kinds in functions.items()},
                          **{struct: SimpleNamespace for _, struct in EXTRACTORS.values()})
    lib.association_calls = association_calls
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
    reader.hits(1, 'gaushit')
    reader.hits(1, 'gaushit')
    reader.hits(0, 'gaushit')
    # indexing leaves each file at its end, so the first read of a file rewinds;
    # then one jump per change of entry
    assert FakeEvent.log == [('toBegin', 'b.root'), ('toBegin', 'a.root'),
                             ('goToEntry', 'a.root', 1), ('goToEntry', 'a.root', 0)]


def test_navigation_without_go_to_entry():
    FakeEvent.log.clear()
    reader = GalleryReader(['a.root'], root=make_root(has_go_to_entry=False))
    assert len(reader.hits(1, 'gaushit')) == 5
    assert len(reader.hits(0, 'gaushit')) == 3
    assert FakeEvent.log == [('toBegin', 'a.root'), ('toBegin', 'a.root')]
    assert len(reader.hits(1, 'gaushit')) == 5


def test_empty_columns():
    empty = SimpleNamespace(channel=[], n_ticks=[], roi_row=[], roi_start=[], roi_offset=[], values=[])
    assert len(columns_to_container('wires', empty)) == 0
    raw = SimpleNamespace(channel=[], pedestal=[], n_samples=[], adc=[], width=0)
    assert columns_to_container('raw_digits', raw).adc.shape == (0, 0)


def test_tracks_and_associations():
    FakeEvent.log.clear()
    root = make_root()
    reader = GalleryReader(['c.root'], root=root)
    tracks = reader.get(0, 'tracks', 'pandoraTrack')
    assert tracks.points(0).tolist() == [[0, 0, 0]] and tracks.valid.tolist() == [True, False]
    assns = reader.get(0, 'track_hits', 'pandoraTrack')
    assert assns.right_of(0).tolist() == [0, 1]
    assert assns.product_tags.tolist() == ['pandoraTrack::Reco2', 'gaushit::Reco1']
    # the stored type and the file's other products are passed to titusio
    ((tag, cpp_type, candidates),) = root.titusio.association_calls
    assert cpp_type == 'art::Assns<recob::Hit,recob::Track,recob::TrackHitMeta>'
    assert sorted(candidates) == ['gaushit::Reco1', 'pandoraTrack::Reco2']
