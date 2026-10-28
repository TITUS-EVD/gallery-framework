"""
Production reader: art-root files through gallery, with products extracted
by the compiled titusio library (UserDev/EventDisplay/Reader). Needs ROOT,
gallery and the product libraries, but no LArSoft services.
"""

import numpy as np

from .base import EventReader, ProductNotFound
from .products import KINDS, EventID, Hits, RawDigits, WireSignals, parse_branch_name

# kind -> (titusio function, titusio column struct)
EXTRACTORS = {
    'hits': ('readHits', 'HitColumns'),
    'wires': ('readWires', 'WireColumns'),
    'channel_rois': ('readChannelROIs', 'WireColumns'),
    'raw_digits': ('readRawDigits', 'RawDigitColumns'),
}


def to_numpy(vector, dtype):
    """
    Copy of a std::vector as a numpy array. PyROOT exposes std::vector of
    numbers through the array interface; anything else is iterated.
    """
    if len(vector) == 0:
        return np.zeros(0, dtype=dtype)
    return np.array(np.asarray(vector), dtype=dtype, copy=True)


def columns_to_container(kind, columns):
    """ a titusio column struct (filled) to the products.py container """
    cls = KINDS[kind].container
    arrays = {name: to_numpy(getattr(columns, name), dtype) for name, dtype in cls.DTYPES.items()}
    if cls is RawDigits:
        arrays['adc'] = arrays['adc'].reshape(len(arrays['channel']), int(columns.width))
        return RawDigits(**arrays)
    if cls is WireSignals and len(arrays['roi_offset']) == 0:
        arrays['roi_offset'] = np.zeros(1, dtype='i8')
    return cls.from_arrays(arrays)


def _import_root():
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    return ROOT


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
        self._columns = {}
        for i, path in enumerate(self._files):
            ev = self._ROOT.gallery.Event(self._ROOT.std.vector['string']([path]))
            self._handles.append(ev)
            self._file_tags.append(self._product_tags(ev))
            local = 0
            while not ev.atEnd():
                aux = ev.eventAuxiliary()
                self._events.append(EventID(int(aux.run()), int(aux.subRun()), int(aux.event())))
                self._entries.append((i, local))
                local += 1
                ev.next()

    @staticmethod
    def _product_tags(ev):
        tags = {}
        for branch in ev.getTFile().Get('Events').GetListOfBranches():
            parsed = parse_branch_name(branch.GetName())
            if parsed is not None and parsed[0] is not None:
                tags.setdefault(parsed[0], []).append(parsed[1])
        return {kind: sorted(t) for kind, t in tags.items()}

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
        function, struct = EXTRACTORS[kind]
        if kind not in self._columns:
            # reused between events: the conversion copies
            self._columns[kind] = getattr(self._lib, struct)()
        columns = self._columns[kind]
        if not getattr(self._lib, function)(self.event_handle(entry), tag, columns):
            raise ProductNotFound(f'{kind} {tag} not in event {self._events[entry]}')
        return columns_to_container(kind, columns)
