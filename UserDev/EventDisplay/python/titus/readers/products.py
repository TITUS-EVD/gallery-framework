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
    waveform n_ticks long. ROI j belongs to row roi_row[j], starts at tick
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


class ProductKind(NamedTuple):
    name: str
    cpp_type: str
    container: type


KINDS = {k.name: k for k in [
    ProductKind('hits', 'recob::Hit', Hits),
    ProductKind('wires', 'recob::Wire', WireSignals),
    ProductKind('channel_rois', 'recob::ChannelROI', WireSignals),
    ProductKind('raw_digits', 'raw::RawDigit', RawDigits),
]}


def parse_branch_name(branch):
    """
    Split an art product branch name, e.g. 'recob::Hits_gaushit__Reco1.', into
    (kind name or None, input tag 'label:instance:process'). art forbids
    underscores in labels, instances and process names. Returns None for
    branches that aren't products (e.g. EventAuxiliary).
    """
    parts = branch.rstrip('.').rsplit('_', 3)
    if len(parts) != 4:
        return None
    friendly, label, instance, process = parts
    kind = next((k.name for k in KINDS.values() if friendly == k.cpp_type + 's'), None)
    return kind, f'{label}:{instance}:{process}'
