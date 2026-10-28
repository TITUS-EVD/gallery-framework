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
