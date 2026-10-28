"""
Interface between TITUS and event sources. The Drawable classes use readers for
products by kind (see products.KINDS) and input tag and get these containers back.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .products import KINDS, EventID, Hits, RawDigits, WireSignals


class ProductNotFound(LookupError):
    pass


def resolve_tag(tag, available):
    """
    Match a full 'label:instance:process' tag, or a shorter 'label' or
    'label:instance', against the available full tags. As in art, a bare
    label means the empty instance.
    """
    if tag in available:
        return tag
    parts = tag.split(':')
    if len(parts) > 3:
        raise ValueError(f'invalid input tag {tag!r}')
    if len(parts) == 1:
        parts.append('')
    matches = [t for t in available if t.split(':')[:len(parts)] == parts]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ProductNotFound(f'no product {tag!r}; available: {", ".join(available) or "none"}')
    raise ProductNotFound(f'{tag!r} is ambiguous, specify one of {", ".join(matches)}')


class ProductCache:
    """
    Products read in the current event, shared by every drawable, so that a
    product (e.g. hits, read by hits, tracks and their associations) is read
    once per event, and redrawing after a GUI change doesn't read it again.
    The owner calls new_event() when the event changes and new_file() when
    the file does. Containers are shared: drawables must not modify them.
    """

    def __init__(self):
        self._products = {}
        self._file = {}

    def new_event(self):
        self._products.clear()

    def new_file(self):
        self._products.clear()
        self._file.clear()

    def get(self, key, read):
        """ the product for key, from read() the first time; a ProductNotFound is remembered too """
        if key not in self._products:
            try:
                self._products[key] = read()
            except ProductNotFound as e:
                self._products[key] = e
        value = self._products[key]
        if isinstance(value, ProductNotFound):
            raise ProductNotFound(*value.args)
        return value

    def file_value(self, key, compute):
        """ something about the current file (e.g. its product list), computed once per file """
        if key not in self._file:
            self._file[key] = compute()
        return self._file[key]


class EventReader(ABC):

    @property
    @abstractmethod
    def events(self) -> List[EventID]:
        """ events in the order they can be read; read methods take an index into this """

    @abstractmethod
    def tags(self, kind, entry=None) -> List[str]:
        """ full input tags of a product kind, in one event or in any event """

    @abstractmethod
    def _read(self, entry, kind, tag):
        """ the product container for a full tag known to exist in this event """

    @property
    def meta(self) -> Dict[str, Any]:
        """ source description: input file, detector, ... """
        return {}

    def entry_of(self, run, subrun, event):
        """ index of an event in events """
        try:
            return self.events.index(EventID(run, subrun, event))
        except ValueError:
            raise LookupError(f'event {run}:{subrun}:{event} not found') from None

    def get(self, entry, kind, tag):
        if kind not in KINDS:
            raise ValueError(f'unknown product kind {kind!r}, expected one of {sorted(KINDS)}')
        if not 0 <= entry < len(self.events):
            raise IndexError(f'event entry {entry} out of range (0-{len(self.events) - 1})')
        return self._read(entry, kind, resolve_tag(tag, self.tags(kind, entry)))

    def hits(self, entry, tag) -> Hits:
        return self.get(entry, 'hits', tag)

    def wires(self, entry, tag) -> WireSignals:
        return self.get(entry, 'wires', tag)

    def channel_rois(self, entry, tag) -> WireSignals:
        return self.get(entry, 'channel_rois', tag)

    def raw_digits(self, entry, tag) -> RawDigits:
        return self.get(entry, 'raw_digits', tag)
