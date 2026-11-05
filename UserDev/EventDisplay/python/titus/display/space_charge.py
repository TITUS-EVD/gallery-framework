"""
Space charge for drawing simulated particles where their charge is.

The simulation moves ionization charge by space charge (a map of
displacements, SpaceChargeService) before it drifts, so hits and wires show
it displaced, while MC truth (sim::MCTrack steps, the interaction vertex) is
where the charge was deposited; Pandora's SCE-corrected products
(pandoraSCE*) are moved back there too. Drawn as stored, both sit off their
charge by up to about a centimetre (a few wires and ticks). Moving them by
the map's forward displacement draws them on it.

Maps are converted from the ROOT files the simulation used by
tools/dump_sce_map.py into titus/detectors/sce/<map name>.npz, and found by
the name a file's configuration gives (titus.readers.file_config).
"""

import dataclasses
import functools
import json
import pathlib

import numpy as np

from ..readers.products import MCTracks, MCTruths, Points, Showers, Tracks

SCE_DIR = pathlib.Path(__file__).parent.parent / 'detectors' / 'sce'

# the position columns of each container moved by the map
POSITION_COLUMNS = {
    Points: ('x', 'y', 'z'),
    Tracks: ('x', 'y', 'z'),
    MCTracks: ('x', 'y', 'z'),
    Showers: ('start_x', 'start_y', 'start_z'),
    MCTruths: ('vx', 'vy', 'vz'),
}

# product kinds in true (deposited) space: simulated ones, and the reco ones
# of producers that corrected them for space charge
SIMULATED_KINDS = ('mctracks', 'mctruths')
SCE_CORRECTED_KINDS = ('tracks', 'showers', 'vertices', 'spacepoints')
SCE_CORRECTED_LABEL_PREFIX = 'pandoraSCE'


class SpaceChargeMap:
    '''
    A voxelized map of forward displacements (cm, at the voxel centers), one
    per side of a dual map ('E' for x < 0, 'W' for x >= 0) or 'all',
    interpolated trilinearly between voxel centers as TH3::Interpolate.
    '''

    def __init__(self, name, sides, axes, meta=None):
        self.name = name
        self.meta = dict(meta or {})
        # {side: (3, nx, ny, nz) array of dx, dy, dz}
        self._sides = {side: np.stack([np.asarray(d, dtype=float) for d in columns]) for side, columns in sides.items()}
        self._centers = [lo + (np.arange(int(n)) + 0.5) * (hi - lo) / int(n) for n, lo, hi in axes]

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            meta = json.loads(bytes(z['meta']))
            sides = {side: [z[f'{side}/d{a}'] for a in 'xyz'] for side in meta['sides']}
            return cls(meta['name'], sides, z['axes'], meta)

    def _interpolate(self, values, points):
        ''' values (3, nx, ny, nz) at points (N, 3) -> (N, 3) '''
        index, weight = [], []
        for c, q in zip(self._centers, points.T):
            i = np.clip(np.searchsorted(c, q) - 1, 0, len(c) - 2)
            index.append(i)
            weight.append(np.clip((q - c[i]) / (c[i + 1] - c[i]), 0, 1))
        (i, j, k), (tx, ty, tz) = index, weight
        out = np.zeros((len(points), 3))
        for a in (0, 1):
            for b in (0, 1):
                for d in (0, 1):
                    w = (tx if a else 1 - tx) * (ty if b else 1 - ty) * (tz if d else 1 - tz)
                    out += w[:, None] * values[:, i + a, j + b, k + d].T
        return out

    def displacement(self, xyz):
        ''' the forward displacement (N, 3) at points (N, 3) '''
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        out = np.zeros_like(xyz)
        if 'all' in self._sides:
            parts = [('all', np.ones(len(xyz), dtype=bool))]
        else:
            parts = [('E', xyz[:, 0] < 0), ('W', xyz[:, 0] >= 0)]
        for side, rows in parts:
            if rows.any():
                out[rows] = self._interpolate(self._sides[side], xyz[rows])
        return out

    def displace(self, xyz):
        '''
        Where the simulation moves charge deposited at xyz (N, 3). Points
        beyond the outermost voxel centers take the edge voxels' values:
        Pandora's corrected points just outside the active volume are charge
        that ended up inside, and match its hits best that way.
        '''
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        return xyz + self.displacement(xyz)


def available_sce_maps(directory=SCE_DIR):
    ''' space charge maps included with TITUS, as {name: path} '''
    return {p.stem: p for p in sorted(pathlib.Path(directory).glob('*.npz'))}


@functools.lru_cache(maxsize=4)
def load_sce_map(name):
    ''' a shipped map by name (e.g. 'SCEoffsets_SBND_E500_dualmap_CV_voxelTH3'), None if TITUS has none '''
    path = available_sce_maps().get(name)
    return None if path is None else SpaceChargeMap.load(path)


def in_true_space(kind, tag):
    '''
    Whether objects of a product kind and tag (e.g. 'tracks',
    'pandoraSCETrack::Reco2') are where charge was deposited, not where the
    simulation moved it: MC truth, and Pandora's SCE-corrected objects
    '''
    if kind in SIMULATED_KINDS:
        return True
    return kind in SCE_CORRECTED_KINDS and str(tag).split(':')[0].startswith(SCE_CORRECTED_LABEL_PREFIX)


def displaced(container, sce_map):
    ''' a copy of a container with its positions moved by the map (the container itself is shared, e.g. cached) '''
    columns = POSITION_COLUMNS.get(type(container))
    if columns is None or len(getattr(container, columns[0])) == 0:
        return container
    xyz = np.stack([getattr(container, c) for c in columns], axis=1)
    moved = sce_map.displace(xyz)
    return dataclasses.replace(container, **{c: moved[:, a] for a, c in enumerate(columns)})
