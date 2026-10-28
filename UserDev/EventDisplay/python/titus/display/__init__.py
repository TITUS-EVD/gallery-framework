"""
Display coordinates for TITUS views, computed from geometry and reader
products without Qt or ROOT.
"""

from .layout import Block, ViewLayout
from .tpc import HitPositions, hit_positions, image_n_ticks, plane_index_table, wire_image
