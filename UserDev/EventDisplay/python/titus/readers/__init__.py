"""
Event sources for TITUS. These readers return products as numpy containers
(products.py). The DumpReader class reads product dumps without ROOT;
GalleryReader reads art-root files through gallery.
"""

from .base import EventReader, ProductNotFound, resolve_tag
from .dump import DumpReader, DumpWriter
from .gallery import GalleryReader
from .products import (KINDS, Associations, EventID, Hits, Points, RawDigits, Showers, Tracks,
                       WireSignals, parse_branch_name)
