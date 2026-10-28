"""
Event sources for TITUS. These readers return products as numpy containers
(products.py). The DumpReader class reads product dumps without ROOT;
GalleryReader reads art-root files through gallery.
"""

from .base import EventReader, ProductNotFound, resolve_tag
from .dump import DumpReader, DumpWriter
from .gallery import GalleryReader
from .products import (ASSOCIATIONS, KINDS, Associations, EventID, Hits, PFParticles, Points, RawDigits,
                       Showers, T0s, Tracks, WireSignals, parse_branch_name)
