"""
Unified View class for wire/tick data in a TPC view, shared by wire images,
hits and projected 3D objects.

A view shows one plane number of one cryostat, including the parallel wire
planes of every TPC in the cryostat. This covers planes which are:
- along the time axis (one block per drift volume) separated by the cathode
  gap. Blocks drifting opposite to TPC 0 are flipped in time
- along the wire axis (split wires), one block per TPC of a drift volume,
  ordered by z and also separated by the cathode gap

View coordinates are pixel units of the view image: pixel (i, j) covers
[i, i + 1) x [j, j + 1). Wire w is centered at w + 0.5, matching LArSoft's
continuous wire coordinate (wire w covers [w - 0.5, w + 0.5)). Times are
continuous ticks as in recob::Hit::PeakTime, where sample t covers
[t, t + 1).
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Block:
    plane_index: int  # TITUS plane index (Geometry.getPlaneID)
    wire_offset: int
    time_offset: int
    n_wires: int
    flip: bool


class ViewLayout:

    def __init__(self, geom, plane, cryostat=0, n_ticks=None):
        """
        n_ticks: samples per waveform. Defaults to the geometry's time range;
        pass the waveform length of the data drawn to fit it exactly.
        """
        self.geom = geom
        self.plane = plane
        self.cryostat = cryostat
        self.n_ticks = int(n_ticks if n_ticks is not None else geom.tRange())
        self.gap = int(geom.cathodeGap())

        first = geom.getPlaneID(plane=plane, tpc=0, cryo=cryostat)
        volumes = geom.detector.cryostats[cryostat].drift_volumes()
        position = {t.id.tpc: (v, z) for v, volume in enumerate(volumes) for z, t in enumerate(volume)}
        blocks = []
        for index in [first] + geom.getOtherPlanes(first):
            plane_geo = geom.detector.planes[index]
            volume, z = position[plane_geo.id.tpc]
            blocks.append(Block(plane_index=index,
                                wire_offset=z * (plane_geo.n_wires + self.gap),
                                time_offset=volume * (self.n_ticks + self.gap),
                                n_wires=plane_geo.n_wires,
                                flip=geom.flipPlane(index)))
        self.blocks: List[Block] = blocks
        self._by_plane = {b.plane_index: b for b in blocks}

    @property
    def shape(self):
        """ (wires, ticks) of the view image """
        return (max(b.wire_offset + b.n_wires for b in self.blocks),
                max(b.time_offset for b in self.blocks) + self.n_ticks)

    def block(self, plane_index):
        return self._by_plane[plane_index]

    def contains(self, plane_index):
        return plane_index in self._by_plane

    def wire_to_view(self, plane_index, wire):
        """ view x of the center of a wire """
        return np.asarray(wire, dtype=float) + self.block(plane_index).wire_offset + 0.5

    def tick_to_view(self, plane_index, tick):
        """
        view y of a continuous time in ticks. Flipped blocks run from the
        end of the block backwards.
        """
        b = self.block(plane_index)
        y = np.asarray(tick, dtype=float)
        if b.flip:
            y = self.n_ticks - y
        return y + b.time_offset

    def block_slices(self, plane_index):
        """ (rows, columns) slices of the view image covered by a block """
        b = self.block(plane_index)
        return (slice(b.wire_offset, b.wire_offset + b.n_wires),
                slice(b.time_offset, b.time_offset + self.n_ticks))
