"""
Columnar containers for the art products TITUS draws. Readers return these,
so drawables don't touch ROOT objects. Row i holds element i of the product
vector, so associations can refer to rows by index.
"""

from dataclasses import dataclass, fields
from typing import ClassVar, Dict, Iterator, NamedTuple, Tuple

import numpy as np


class EventID(NamedTuple):
    run: int
    subrun: int
    event: int


class Columns:
    """ Base for containers whose dataclass fields are all numpy arrays """
    DTYPES: ClassVar[Dict[str, str]] = {}

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, np.asarray(getattr(self, f.name), dtype=self.DTYPES[f.name]))

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_arrays(cls, arrays):
        missing = set(f.name for f in fields(cls)) - set(arrays)
        if missing:
            raise ValueError(f'{cls.__name__}: missing columns {sorted(missing)}')
        return cls(**{f.name: arrays[f.name] for f in fields(cls)})

    @classmethod
    def empty(cls):
        return cls(**{f.name: [] for f in fields(cls)})


@dataclass(frozen=True)
class Hits(Columns):
    """ recob::Hit """
    DTYPES: ClassVar[Dict[str, str]] = {
        'channel': 'u4', 'cryostat': 'i4', 'tpc': 'i4', 'plane': 'i4', 'wire': 'i4',
        'peak_time': 'f4', 'rms': 'f4', 'integral': 'f4', 'peak_amplitude': 'f4',
        'start_tick': 'i4', 'end_tick': 'i4',
    }
    channel: np.ndarray
    cryostat: np.ndarray
    tpc: np.ndarray
    plane: np.ndarray
    wire: np.ndarray
    peak_time: np.ndarray
    rms: np.ndarray
    integral: np.ndarray
    peak_amplitude: np.ndarray
    start_tick: np.ndarray
    end_tick: np.ndarray

    def __len__(self):
        return len(self.channel)

    def on_plane(self, cryostat, tpc, plane) -> np.ndarray:
        """ boolean mask of the hits on this plane """
        return (self.cryostat == cryostat) & (self.tpc == tpc) & (self.plane == plane)


@dataclass(frozen=True)
class WireSignals(Columns):
    """
    recob::Wire or recob::ChannelROI: per channel, regions of interest of a
    waveform n_ticks long. n_ticks is the stored length: the readout window
    for recob::Wire, but where the last ROI ends for ChannelROI. ROI j belongs to row roi_row[j], starts at tick
    roi_start[j] and holds values[roi_offset[j]:roi_offset[j + 1]].
    """
    DTYPES: ClassVar[Dict[str, str]] = {
        'channel': 'u4', 'n_ticks': 'i4',
        'roi_row': 'i4', 'roi_start': 'i4', 'roi_offset': 'i8', 'values': 'f4',
    }
    channel: np.ndarray
    n_ticks: np.ndarray
    roi_row: np.ndarray
    roi_start: np.ndarray
    roi_offset: np.ndarray
    values: np.ndarray

    def __len__(self):
        return len(self.channel)

    @classmethod
    def empty(cls):
        return cls(channel=[], n_ticks=[], roi_row=[], roi_start=[], roi_offset=[0], values=[])

    @classmethod
    def from_rois(cls, channels, n_ticks, rois):
        """
        channels, n_ticks: per row. rois: per row, a list of (start tick, values).
        """
        roi_row, roi_start, chunks = [], [], []
        for row, row_rois in enumerate(rois):
            for start, values in row_rois:
                roi_row.append(row)
                roi_start.append(start)
                chunks.append(np.asarray(values, dtype='f4'))
        offsets = np.concatenate([[0], np.cumsum([len(c) for c in chunks], dtype='i8')])
        values = np.concatenate(chunks) if chunks else np.zeros(0, dtype='f4')
        return cls(channel=channels, n_ticks=n_ticks, roi_row=roi_row, roi_start=roi_start,
                   roi_offset=offsets, values=values)

    def rois(self, row) -> Iterator[Tuple[int, np.ndarray]]:
        """ (start tick, values) of each ROI of a row """
        for j in np.flatnonzero(self.roi_row == row):
            yield int(self.roi_start[j]), self.values[self.roi_offset[j]:self.roi_offset[j + 1]]

    def dense(self, row) -> np.ndarray:
        """ the full waveform of a row, zero outside the ROIs """
        out = np.zeros(int(self.n_ticks[row]), dtype='f4')
        for start, values in self.rois(row):
            out[start:start + len(values)] = values
        return out


@dataclass(frozen=True)
class RawDigits(Columns):
    """
    raw::RawDigit, uncompressed. adc rows are padded with zeros to the
    longest waveform; n_samples holds each row's length.
    """
    DTYPES: ClassVar[Dict[str, str]] = {'channel': 'u4', 'pedestal': 'f4', 'n_samples': 'i4', 'adc': 'i2'}
    channel: np.ndarray
    pedestal: np.ndarray
    n_samples: np.ndarray
    adc: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        if self.adc.ndim != 2:
            object.__setattr__(self, 'adc', self.adc.reshape(len(self.channel), -1))

    def __len__(self):
        return len(self.channel)


@dataclass(frozen=True)
class Tracks(Columns):
    """
    recob::Track trajectories: track i has points point_offset[i] to
    point_offset[i + 1]; valid marks points with a valid position.
    """
    DTYPES: ClassVar[Dict[str, str]] = {
        'id': 'i4', 'point_offset': 'i8', 'x': 'f4', 'y': 'f4', 'z': 'f4', 'valid': '?',
    }
    id: np.ndarray
    point_offset: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    valid: np.ndarray

    def __len__(self):
        return len(self.id)

    @classmethod
    def empty(cls):
        return cls(id=[], point_offset=[0], x=[], y=[], z=[], valid=[])

    @classmethod
    def from_points(cls, ids, points, valid=None):
        """ points: per track an (n, 3) array; valid: per track (n,) booleans """
        points = [np.asarray(p, dtype='f4').reshape(-1, 3) for p in points]
        valid = [np.ones(len(p), dtype=bool) for p in points] if valid is None else valid
        offsets = np.concatenate([[0], np.cumsum([len(p) for p in points], dtype='i8')])
        xyz = np.concatenate(points) if points else np.zeros((0, 3), dtype='f4')
        return cls(id=ids, point_offset=offsets, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                   valid=np.concatenate(valid) if valid else [])

    def points(self, row, valid_only=True) -> np.ndarray:
        """ (n, 3) trajectory points of a track """
        sl = slice(self.point_offset[row], self.point_offset[row + 1])
        xyz = np.stack([self.x[sl], self.y[sl], self.z[sl]], axis=1)
        return xyz[self.valid[sl]] if valid_only else xyz


@dataclass(frozen=True)
class Points(Columns):
    """ recob::Vertex or recob::SpacePoint positions """
    DTYPES: ClassVar[Dict[str, str]] = {'id': 'i4', 'x': 'f4', 'y': 'f4', 'z': 'f4'}
    id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __len__(self):
        return len(self.id)

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.x, self.y, self.z], axis=1)


@dataclass(frozen=True)
class Showers(Columns):
    """ recob::Shower: start point, unit direction, length and opening angle """
    DTYPES: ClassVar[Dict[str, str]] = {
        'id': 'i4', 'start_x': 'f4', 'start_y': 'f4', 'start_z': 'f4',
        'dir_x': 'f4', 'dir_y': 'f4', 'dir_z': 'f4', 'length': 'f4', 'open_angle': 'f4',
        'best_plane': 'i4',
    }
    id: np.ndarray
    start_x: np.ndarray
    start_y: np.ndarray
    start_z: np.ndarray
    dir_x: np.ndarray
    dir_y: np.ndarray
    dir_z: np.ndarray
    length: np.ndarray
    open_angle: np.ndarray
    best_plane: np.ndarray

    def __len__(self):
        return len(self.id)


@dataclass(frozen=True)
class Associations(Columns):
    """
    art::Assns: row j links element left[j] of product product_tags[left_product[j]]
    to element right[j] of product product_tags[right_product[j]]. An empty
    tag means the product couldn't be identified. meta_index is the
    recob::TrackHitMeta index (the trajectory point of the hit), or -1.
    """
    DTYPES: ClassVar[Dict[str, str]] = {
        'left': 'i4', 'right': 'i4', 'left_product': 'i4', 'right_product': 'i4',
        'meta_index': 'i4', 'product_tags': 'U',
    }
    left: np.ndarray
    right: np.ndarray
    left_product: np.ndarray
    right_product: np.ndarray
    meta_index: np.ndarray
    product_tags: np.ndarray

    def __len__(self):
        return len(self.left)

    def right_of(self, left_row) -> np.ndarray:
        """ right elements associated with a left element """
        return self.right[self.left == left_row]


@dataclass(frozen=True)
class PFParticles(Columns):
    """ recob::PFParticle: Self() index, PDG code and parent (-1 for primaries) """
    DTYPES: ClassVar[Dict[str, str]] = {'self_index': 'i4', 'pdg': 'i4', 'parent': 'i4'}
    self_index: np.ndarray
    pdg: np.ndarray
    parent: np.ndarray

    def __len__(self):
        return len(self.self_index)


@dataclass(frozen=True)
class T0s(Columns):
    """ anab::T0: time in ns, trigger type and ID """
    DTYPES: ClassVar[Dict[str, str]] = {'time': 'f8', 'trigger_type': 'i4', 'id': 'i4'}
    time: np.ndarray
    trigger_type: np.ndarray
    id: np.ndarray

    def __len__(self):
        return len(self.time)


class ProductKind(NamedTuple):
    name: str
    container: type
    # C++ product types read as this kind
    types: Tuple[str, ...]


# association kinds: (left type, right type, metadata types). art may store
# either order; readers always return left = the first type here
ASSOCIATIONS = {
    'track_hits': ('recob::Track', 'recob::Hit', ('recob::TrackHitMeta', 'void')),
    'pfp_tracks': ('recob::PFParticle', 'recob::Track', ('void',)),
    'pfp_t0s': ('recob::PFParticle', 'anab::T0', ('void',)),
    'pfp_vertices': ('recob::PFParticle', 'recob::Vertex', ('void',)),
    'pfp_spacepoints': ('recob::PFParticle', 'recob::SpacePoint', ('void',)),
    'spacepoint_hits': ('recob::SpacePoint', 'recob::Hit', ('void',)),
}


def _association_types(left, right, metas):
    return tuple(f'art::Assns<{a},{b},{d}>' for d in metas for a, b in ((left, right), (right, left)))


def association_is_swapped(kind, cpp_type):
    """ True if an association product type has the kind's two types in reverse order """
    left = ASSOCIATIONS[kind][0]
    return not cpp_type.startswith(f'art::Assns<{left},')


KINDS = {k.name: k for k in [
    ProductKind('hits', Hits, ('std::vector<recob::Hit>',)),
    ProductKind('wires', WireSignals, ('std::vector<recob::Wire>',)),
    ProductKind('channel_rois', WireSignals, ('std::vector<recob::ChannelROI>',)),
    ProductKind('raw_digits', RawDigits, ('std::vector<raw::RawDigit>',)),
    ProductKind('tracks', Tracks, ('std::vector<recob::Track>',)),
    ProductKind('vertices', Points, ('std::vector<recob::Vertex>',)),
    ProductKind('spacepoints', Points, ('std::vector<recob::SpacePoint>',)),
    ProductKind('showers', Showers, ('std::vector<recob::Shower>',)),
    ProductKind('pfparticles', PFParticles, ('std::vector<recob::PFParticle>',)),
    ProductKind('t0s', T0s, ('std::vector<anab::T0>',)),
] + [ProductKind(kind, Associations, _association_types(*types)) for kind, types in ASSOCIATIONS.items()]}


def friendly_name(cpp_type):
    """
    art's branch name prefix for a product type: 'std::vector<recob::Hit>' ->
    'recob::Hits', 'art::Assns<A,B,D>' -> 'ABDart::Assns'
    """
    if cpp_type.startswith('std::vector<'):
        return cpp_type[len('std::vector<'):-1] + 's'
    if cpp_type.startswith('art::Assns<'):
        return ''.join(cpp_type[len('art::Assns<'):-1].split(',')) + 'art::Assns'
    raise ValueError(f'no friendly name rule for {cpp_type}')


_BY_FRIENDLY_NAME = {friendly_name(t): (k.name, t) for k in KINDS.values() for t in k.types}


def parse_branch(branch):
    """
    Split an art product branch name, e.g. 'recob::Hits_gaushit__Reco1.', into
    (kind name, input tag 'label:instance:process', C++ product type); kind and
    type are None for products TITUS doesn't read. art forbids underscores in
    labels, instances and process names. Returns None for branches that
    aren't products (e.g. EventAuxiliary).
    """
    parts = branch.rstrip('.').rsplit('_', 3)
    if len(parts) != 4:
        return None
    friendly, label, instance, process = parts
    kind, cpp_type = _BY_FRIENDLY_NAME.get(friendly, (None, None))
    return kind, f'{label}:{instance}:{process}', cpp_type


def parse_branch_name(branch):
    """ (kind name or None, input tag) of a product branch, see parse_branch """
    parsed = parse_branch(branch)
    return None if parsed is None else parsed[:2]
