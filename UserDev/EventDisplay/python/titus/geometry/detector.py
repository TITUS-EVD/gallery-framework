"""
Detector description loaded from a JSON dump of the LArSoft services, see
tools/dump_detector.py

Units in LArSoft convention: cm, microseconds, ticks. 
Identifiers follow LArSoft ordering: (cryostat, tpc, plane, wire).
"""

import gzip
import json
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

SUPPORTED_SCHEMA_VERSIONS = (1,)

# two wire directions are parallel if their angles to z agree (mod pi) to this
PARALLEL_TOLERANCE = 1e-6


class PlaneID(NamedTuple):
    cryostat: int
    tpc: int
    plane: int


class TPCID(NamedTuple):
    cryostat: int
    tpc: int


@dataclass(frozen=True)
class Box:
    min: np.ndarray
    max: np.ndarray

    @classmethod
    def from_json(cls, d):
        return cls(np.asarray(d['min'], dtype=float), np.asarray(d['max'], dtype=float))

    def contains(self, points):
        """ points: (3,) or (N, 3). Boundaries are inclusive, as in LArSoft """
        points = np.asarray(points, dtype=float)
        return np.all((points >= self.min) & (points <= self.max), axis=-1)


@dataclass(frozen=True)
class Timing:
    tpc_tick_period: float
    optical_tick_period: float
    trigger_offset_tpc: float
    trigger_offset_tpc_ticks: int
    trigger_time: float
    beam_gate_time: float
    g4_to_elec_time_at_zero: float
    drift_velocity: float
    efield: Tuple[float, ...]
    temperature: float
    number_time_samples: int
    readout_window_size: int
    tpc_frame_ticks: int
    optical_frame_ticks: int

    @property
    def time_to_cm(self):
        """ drift distance per TPC tick """
        return self.tpc_tick_period * self.drift_velocity

    @classmethod
    def from_json(cls, d):
        return cls(
            tpc_tick_period=d['tpc_tick_period'],
            optical_tick_period=d['optical_tick_period'],
            trigger_offset_tpc=d['trigger_offset_tpc'],
            trigger_offset_tpc_ticks=d['trigger_offset_tpc_ticks'],
            trigger_time=d['trigger_time'],
            beam_gate_time=d['beam_gate_time'],
            g4_to_elec_time_at_zero=d['g4_to_elec_time_at_zero'],
            drift_velocity=d['drift_velocity'],
            efield=tuple(d['efield']),
            temperature=d['temperature'],
            number_time_samples=d['number_time_samples'],
            readout_window_size=d['readout_window_size'],
            tpc_frame_ticks=d['tpc_frame_ticks'],
            optical_frame_ticks=d['optical_frame_ticks'],
        )


@dataclass(frozen=True)
class Plane:
    id: PlaneID
    view: int
    signal_type: int
    center: np.ndarray
    n_wires: int
    wire_pitch: float
    theta_z: float
    first_wire_center: np.ndarray
    increasing_wire_direction: np.ndarray
    x_at_tick0: float
    x_per_tick: float

    @classmethod
    def from_json(cls, d):
        return cls(
            id=PlaneID(*d['id']),
            view=d['view'],
            signal_type=d['signal_type'],
            center=np.asarray(d['center'], dtype=float),
            n_wires=d['n_wires'],
            wire_pitch=d['wire_pitch'],
            theta_z=d['theta_z'],
            first_wire_center=np.asarray(d['first_wire']['center'], dtype=float),
            increasing_wire_direction=np.asarray(d['increasing_wire_direction'], dtype=float),
            x_at_tick0=d['x_at_tick0'],
            x_per_tick=d['x_per_tick'],
        )

    def wire_coordinate(self, points):
        """
        Continuous wire number of 3D points (as geo::PlaneGeo::WireCoordinate).
        points: (3,) or (N, 3).
        """
        offset = np.asarray(points, dtype=float) - self.first_wire_center
        return offset @ self.increasing_wire_direction / self.wire_pitch

    def nearest_wire(self, points):
        """
        Nearest wire number, or -1 if the point is beyond the first/last wire.
        Rounding from LArSoft: truncating coordinate + 0.5 toward zero, so
        coordinates down to -1 map to wire 0
        """
        wire = np.trunc(self.wire_coordinate(points) + 0.5).astype(int)
        return np.where((wire >= 0) & (wire < self.n_wires), wire, -1)

    def x_to_ticks(self, x):
        """ as detinfo::DetectorPropertiesData::ConvertXToTicks """
        return (np.asarray(x, dtype=float) - self.x_at_tick0) / self.x_per_tick

    def ticks_to_x(self, ticks):
        """ as detinfo::DetectorPropertiesData::ConvertTicksToX """
        return self.x_at_tick0 + np.asarray(ticks, dtype=float) * self.x_per_tick

    def is_parallel_to(self, other):
        delta = (self.theta_z - other.theta_z) % np.pi
        return min(delta, np.pi - delta) < PARALLEL_TOLERANCE


@dataclass(frozen=True)
class TPC:
    id: TPCID
    bounds: Box
    active_bounds: Box
    drift_dir: np.ndarray
    drift_distance: float
    planes: Tuple[Plane, ...]

    @classmethod
    def from_json(cls, d):
        return cls(
            id=TPCID(*d['id']),
            bounds=Box.from_json(d['bounds']),
            active_bounds=Box.from_json(d['active_bounds']),
            drift_dir=np.asarray(d['drift_dir'], dtype=float),
            drift_distance=d['drift_distance'],
            planes=tuple(Plane.from_json(p) for p in d['planes']),
        )


@dataclass(frozen=True)
class Cryostat:
    id: int
    bounds: Box
    half_width: float
    half_height: float
    length: float
    tpcs: Tuple[TPC, ...]

    @classmethod
    def from_json(cls, d):
        return cls(id=d['id'], bounds=Box.from_json(d['bounds']),
                   half_width=d['half_width'], half_height=d['half_height'], length=d['length'],
                   tpcs=tuple(TPC.from_json(t) for t in d['tpcs']))

    def drift_volumes(self) -> List[Tuple[TPC, ...]]:
        """
        TPCs grouped by shared drift volume (same x extent). Volumes are
        ordered by their lowest TPC number, TPCs within a volume by z.
        ICARUS splits each drift volume into two TPCs at z = 0.
        """
        volumes = {}
        for t in self.tpcs:
            key = tuple(np.round([t.bounds.min[0], t.bounds.max[0]], 3))
            volumes.setdefault(key, []).append(t)
        return [tuple(sorted(v, key=lambda t: t.bounds.min[2]))
                for v in sorted(volumes.values(), key=lambda v: min(t.id.tpc for t in v))]


@dataclass(frozen=True)
class ChannelMap:
    """ One row per (channel, wire) pair; a channel can read out several wires """
    n_channels: int
    channel: np.ndarray
    cryostat: np.ndarray
    tpc: np.ndarray
    plane: np.ndarray
    wire: np.ndarray

    @classmethod
    def from_json(cls, d):
        rows = {k: np.asarray(v, dtype=np.int32) for k, v in d['rows'].items()}
        order = np.argsort(rows['channel'], kind='stable')
        return cls(n_channels=d['n_channels'], **{k: v[order] for k, v in rows.items()})

    def wires(self, channel):
        """ (cryostat, tpc, plane, wire) for every wire read out by this channel """
        lo, hi = np.searchsorted(self.channel, [channel, channel + 1])
        return [(int(self.cryostat[i]), int(self.tpc[i]), int(self.plane[i]), int(self.wire[i]))
                for i in range(lo, hi)]


@dataclass(frozen=True)
class OpDets:
    """ Indexed by optical channel """
    opdet: np.ndarray
    center: np.ndarray
    shape: Tuple[str, ...]

    @classmethod
    def from_json(cls, d):
        chans = sorted(d['channels'], key=lambda c: c['channel'])
        return cls(opdet=np.array([c['opdet'] for c in chans], dtype=int),
                   center=np.array([c['center'] for c in chans], dtype=float).reshape(-1, 3),
                   shape=tuple(c['shape'] for c in chans))

    def __len__(self):
        return len(self.opdet)


@dataclass(frozen=True)
class Detector:
    name: str
    gdml: str
    timing: Timing
    cryostats: Tuple[Cryostat, ...]
    channel_map: ChannelMap
    opdets: OpDets
    # CRT modules/strips, kept as dumped until the CRT view is ported
    auxdets: Tuple[dict, ...] = ()
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, d):
        meta = d['meta']
        if meta.get('schema_version') not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported detector schema version {meta.get('schema_version')}")
        return cls(
            name=meta['detector'],
            gdml=meta['gdml'],
            timing=Timing.from_json(d['timing']),
            cryostats=tuple(Cryostat.from_json(c) for c in d['cryostats']),
            channel_map=ChannelMap.from_json(d['channel_map']),
            opdets=OpDets.from_json(d['opdets']),
            auxdets=tuple(d.get('auxdets') or ()),
            meta=meta,
        )

    @classmethod
    def load(cls, path):
        path = str(path)
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rt') as f:
            return cls.from_json(json.load(f))

    # --- lookups ---

    @property
    def tpcs(self) -> List[TPC]:
        return [t for c in self.cryostats for t in c.tpcs]

    @property
    def planes(self) -> List[Plane]:
        return [p for t in self.tpcs for p in t.planes]

    def tpc(self, cryostat, tpc) -> TPC:
        return self.cryostats[cryostat].tpcs[tpc]

    def plane(self, cryostat, tpc, plane) -> Plane:
        return self.cryostats[cryostat].tpcs[tpc].planes[plane]

    def tpc_of(self, points) -> np.ndarray:
        """
        (cryostat, tpc) containing each point, or (-1, -1) outside every TPC.
        points: (3,) -> (2,); (N, 3) -> (N, 2).
        """
        points = np.asarray(points, dtype=float)
        result = np.full(points.shape[:-1] + (2,), -1, dtype=int)
        found = np.zeros(points.shape[:-1], dtype=bool)
        for t in self.tpcs:
            inside = t.bounds.contains(points) & ~found
            result[inside] = t.id
            found |= inside
        return result

    def parallel_planes(self, plane_id) -> List[PlaneID]:
        """ Planes in other TPCs of the same cryostat with wires parallel to this plane's """
        ref = self.plane(*plane_id)
        return [p.id for t in self.cryostats[ref.id.cryostat].tpcs if t.id.tpc != ref.id.tpc
                for p in t.planes if p.is_parallel_to(ref)]
