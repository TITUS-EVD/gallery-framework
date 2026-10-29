"""
Production reader: art-root files through gallery, with products extracted
by the compiled titusio library (UserDev/EventDisplay/Reader). Needs ROOT,
gallery and the product libraries, but no LArSoft services.
"""

import numpy as np

from .base import EventReader, ProductNotFound
from .products import ASSOCIATIONS, KINDS, CRTFEBData, EventID, Hits, RawDigits, WireSignals, parse_branch

# kind -> (titusio function, titusio column struct)
EXTRACTORS = {
    'hits': ('readHits', 'HitColumns'),
    'wires': ('readWires', 'WireColumns'),
    'channel_rois': ('readChannelROIs', 'WireColumns'),
    'raw_digits': ('readRawDigits', 'RawDigitColumns'),
    'tracks': ('readTracks', 'TrackColumns'),
    'vertices': ('readVertices', 'PointColumns'),
    'spacepoints': ('readSpacePoints', 'PointColumns'),
    'showers': ('readShowers', 'ShowerColumns'),
    'pfparticles': ('readPFParticles', 'PFParticleColumns'),
    't0s': ('readT0s', 'T0Columns'),
    'clusters': ('readClusters', 'ClusterColumns'),
    'endpoints2d': ('readEndPoints2D', 'EndPoint2DColumns'),
    'mctruths': ('readMCTruths', 'MCTruthColumns'),
    'mctracks': ('readMCTracks', 'MCTrackColumns'),
    'opflashes': ('readOpFlashes', 'OpFlashColumns'),
    'opdet_waveforms': ('readOpDetWaveforms', 'OpDetWaveformColumns'),
    'crt_febdata': ('readCRTFEBData', 'CRTFEBDataColumns'),
    'daq_timestamps': ('readDAQTimestamps', 'DAQTimestampColumns'),
    **{kind: ('readAssociation', 'AssociationColumns') for kind in ASSOCIATIONS},
}


def to_numpy(vector, dtype):
    """
    Copy of a std::vector as a numpy array. PyROOT exposes std::vector of
    numbers through the array interface; anything else is iterated.
    """
    if len(vector) == 0:
        return np.zeros(0, dtype=dtype)
    if np.dtype(dtype).kind == 'U':
        return np.array([str(v) for v in vector], dtype=dtype)
    array = np.asarray(vector)
    if array.dtype.kind in 'SU':
        # unsigned char elements iterated as one-character strings
        array = np.array([ord(c) for c in vector])
    return np.array(array, dtype=dtype, copy=True)


def columns_to_container(kind, columns):
    """ a titusio column struct (filled) to the products.py container """
    cls = KINDS[kind].container
    arrays = {name: to_numpy(getattr(columns, name), dtype) for name, dtype in cls.DTYPES.items()}
    if cls is RawDigits:
        arrays['adc'] = arrays['adc'].reshape(len(arrays['channel']), int(columns.width))
        return RawDigits(**arrays)
    if cls is CRTFEBData:
        arrays['adc'] = arrays['adc'].reshape(len(arrays['mac5']), CRTFEBData.N_SIPMS)
    if cls is WireSignals and len(arrays['roi_offset']) == 0:
        arrays['roi_offset'] = np.zeros(1, dtype='i8')
    return cls.from_arrays(arrays)


def _import_root():
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    # no PyROOT event polling thread (see evd.py); no effect once ROOT is set up
    ROOT.PyConfig.StartGUIThread = False
    return ROOT


def read_event_product(lib, ev, kind, tag, columns=None, cpp_type=None, candidate_tags=None):
    """
    A product of a positioned gallery.Event, through the titusio library
    (ROOT.titusio). tag: full 'label:instance:process' or as accepted by
    art::InputTag. columns: a reusable titusio column struct.
    Associations also need cpp_type, the product type in the file (e.g.
    from parse_branch), and candidate_tags, a std::vector<std::string> of
    the tags of the products they may point into.
    """
    if kind not in EXTRACTORS:
        raise NotImplementedError(f'{kind} products are not read by the titusio library yet')
    function, struct = EXTRACTORS[kind]
    if columns is None:
        columns = getattr(lib, struct)()
    if kind in ASSOCIATIONS:
        if cpp_type is None or candidate_tags is None:
            raise ValueError(f'{kind}: reading an association needs its cpp_type and candidate_tags')
        found = getattr(lib, function)(ev, tag, cpp_type, candidate_tags, columns)
    else:
        found = getattr(lib, function)(ev, tag, columns)
    if not found:
        raise ProductNotFound(f'{kind} {tag} not in this event')
    return columns_to_container(kind, columns)


def file_products(ev):
    """
    {(kind, tag): C++ product type} of the products TITUS reads in the
    current file of a gallery.Event, from its branch names
    """
    types = {}
    for branch in ev.getTFile().Get('Events').GetListOfBranches():
        parsed = parse_branch(branch.GetName())
        if parsed is None or parsed[0] is None:
            continue
        kind, tag, cpp_type = parsed
        # a producer may store an association with and without metadata
        # (Pandora's track-hit associations): prefer the kind's first type
        previous = types.get((kind, tag))
        if previous is None or KINDS[kind].types.index(cpp_type) < KINDS[kind].types.index(previous):
            types[(kind, tag)] = cpp_type
    return types


def association_candidates(types):
    """ tags of the non-association products, which associations may point into """
    return list(dict.fromkeys(tag for (kind, tag) in types if kind not in ASSOCIATIONS))


def string_vector(values):
    """ a std::vector<std::string>, e.g. association candidate tags """
    return _import_root().std.vector['string'](list(values))


def titusio():
    """ the titusio library, loaded through ROOT """
    return _import_root().titusio


class GalleryReader(EventReader):

    def __init__(self, files, root=None):
        """
        files: art-root paths or xrootd URLs. root: the ROOT module (for tests);
        imported if not given.
        """
        self._ROOT = root or _import_root()
        self._lib = self._ROOT.titusio
        self._files = [str(f) for f in ([files] if isinstance(files, str) else files)]
        self._handles = []
        self._entries = []  # (file index, entry in file)
        self._events = []
        self._file_tags = []  # per file, {kind: [full tags]}
        self._file_types = []  # per file, {(kind, tag): C++ product type}
        self._columns = {}
        for i, path in enumerate(self._files):
            ev = self._ROOT.gallery.Event(self._ROOT.std.vector['string']([path]))
            self._handles.append(ev)
            tags, types = self._product_tags(ev)
            self._file_tags.append(tags)
            self._file_types.append(types)
            local = 0
            while not ev.atEnd():
                aux = ev.eventAuxiliary()
                self._events.append(EventID(int(aux.run()), int(aux.subRun()), int(aux.event())))
                self._entries.append((i, local))
                local += 1
                ev.next()

    @staticmethod
    def _product_tags(ev):
        types = file_products(ev)
        tags = {}
        for kind, tag in types:
            tags.setdefault(kind, []).append(tag)
        return {kind: sorted(t) for kind, t in tags.items()}, types

    @property
    def events(self):
        return self._events

    @property
    def meta(self):
        return {'source_files': list(self._files)}

    def tags(self, kind, entry=None):
        files = range(len(self._files)) if entry is None else [self._entries[entry][0]]
        found = []
        for i in files:
            found += [t for t in self._file_tags[i].get(kind, []) if t not in found]
        return found

    def event_handle(self, entry):
        """ the gallery.Event positioned at an entry """
        file_index, local = self._entries[entry]
        ev = self._handles[file_index]
        if ev.atEnd():
            # past the end there is no current file, and goToEntry throws
            ev.toBegin()
        if ev.eventEntry() != local:
            if hasattr(ev, 'goToEntry'):
                ev.goToEntry(local)
            else:
                if ev.eventEntry() > local:
                    ev.toBegin()
                while ev.eventEntry() != local:
                    ev.next()
        return ev

    def _read(self, entry, kind, tag):
        if kind not in self._columns and kind in EXTRACTORS:
            # reused between events: the conversion copies
            self._columns[kind] = getattr(self._lib, EXTRACTORS[kind][1])()
        cpp_type = candidates = None
        if kind in ASSOCIATIONS:
            file_index = self._entries[entry][0]
            cpp_type = self._file_types[file_index][(kind, tag)]
            # every non-association product of the file: the ones pointed into are among them
            candidates = self._ROOT.std.vector['string'](association_candidates(self._file_types[file_index]))
        try:
            return read_event_product(self._lib, self.event_handle(entry), kind, tag, self._columns.get(kind),
                                      cpp_type, candidates)
        except ProductNotFound:
            raise ProductNotFound(f'{kind} {tag} not in event {self._events[entry]}') from None
