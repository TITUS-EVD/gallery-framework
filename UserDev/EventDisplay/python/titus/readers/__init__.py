"""
Event sources for TITUS. These readers return products as numpy containers
(products.py). The DumpReader class reads product dumps without ROOT.
"""

from .base import EventReader, ProductNotFound, resolve_tag
from .dump import DumpReader, DumpWriter
from .products import KINDS, EventID, Hits, RawDigits, WireSignals, parse_branch_name
