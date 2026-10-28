"""
The geometry interface used by the TITUS GUI and drawables, built from a
Detector description plus display settings (<detector>.display.json) instead
of per-detector subclasses and the LArSoft services.

Method names follow the legacy titus.gallery_interface.geometry classes so
existing callers keep working. Everything that follows from the detector
itself (plane numbering, flips, shifts, plane pairing across TPCs) is computed
from the Detector; the display settings only hold preferences.

TITUS plane index: position of a plane in Detector.planes, i.e. ordered by
(cryostat, tpc, plane).
"""

import copy
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from . import DETECTOR_DIR, load_detector
from .detector import Detector, PlaneID

LOGO_PATH = pathlib.Path(__file__).parent.parent.parent / 'resources' / 'logos'

# op det type when the display settings don't list one per channel
OPDET_NAME_BY_SHAPE = {'TGeoSphere': 'pmt_coated', 'TGeoBBox': 'xarapuca_vis'}


@dataclass
class ServiceHandles:
    """
    LArSoft service providers, still needed by the C++ drawers until they are
    ported. The clock and detector property entries are the per-job data
    objects (DetectorClocks.DataForJob(), DetectorProperties.DataFor(clocks)).
    """
    geometry_core: Any = None
    detector_properties: Any = None
    wire_readout: Any = None
    detector_clocks: Any = None
    lar_properties: Any = None
    aux_det_geometry: Any = None


def load_display_settings(detector_name, directory=DETECTOR_DIR):
    """ common.display.json overlaid with <detector>.display.json (one level deep) """
    settings = {}
    for name in ('common', detector_name):
        path = pathlib.Path(directory) / f'{name}.display.json'
        if not path.exists():
            continue
        with open(path) as f:
            for key, value in json.load(f).items():
                if key.startswith('_'):
                    continue
                if isinstance(value, dict) and isinstance(settings.get(key), dict):
                    settings[key] = {**settings[key], **value}
                else:
                    settings[key] = value
    return settings


def _tuples(obj):
    """ JSON lists back to the tuples pyqtgraph gradient states were written with """
    if isinstance(obj, list):
        return tuple(_tuples(o) for o in obj)
    if isinstance(obj, dict):
        return {k: (_tuples(v) if k == 'ticks' else v) for k, v in obj.items()}
    return obj


class Geometry:

    def __init__(self, detector: Detector, display: Dict[str, Any],
                 services: Optional[ServiceHandles] = None):
        self.detector = detector
        self.display = display
        self.services = services or ServiceHandles()

        timing = detector.timing
        self._time2Cm = timing.time_to_cm
        self._tRange = timing.number_time_samples
        self._readoutWindowSize = timing.number_time_samples

        planes = detector.planes
        self._planes = planes
        self._planeid_map = {(p.id.plane, p.id.tpc, p.id.cryostat): i for i, p in enumerate(planes)}
        self._plane_index = {p.id: i for i, p in enumerate(planes)}

        self._split_wire = bool(display.get('split_wire', False))
        self._cathodeGap = display.get('cathode_gap_cm', 0.) / self._time2Cm

        # legacy code reads these attributes directly
        self._levels = [tuple(l) for l in display.get('levels', [])]
        self._pedestals = list(display.get('pedestals', []))
        self._wRange = [self._wire_range(p) for p in planes]

        self._plane_flip = [self._flip(p.id) for p in planes]
        self._plane_shift = [self._shift(p.id) for p in planes]
        self._plane_mix = self._compute_plane_mix()

    @classmethod
    def load(cls, detector_name, gdml=None, services=None):
        return cls(load_detector(detector_name, gdml), load_display_settings(detector_name), services)

    # --- layout, computed from the detector ---

    def _tpc_of_plane(self, plane_id):
        return self.detector.tpc(plane_id.cryostat, plane_id.tpc)

    def _drift_volume_of(self, plane_id):
        """ (index of the drift volume in its cryostat, position of the TPC in it by z) """
        for v, volume in enumerate(self.detector.cryostats[plane_id.cryostat].drift_volumes()):
            for z, tpc in enumerate(volume):
                if tpc.id.tpc == plane_id.tpc:
                    return v, z
        raise LookupError(plane_id)

    def _flip(self, plane_id):
        """ time axis is flipped for TPCs drifting opposite to TPC 0 of the cryostat """
        reference = self.detector.tpc(plane_id.cryostat, 0)
        return bool(np.sign(self._tpc_of_plane(plane_id).drift_dir[0]) != np.sign(reference.drift_dir[0]))

    def _shift(self, plane_id):
        """ with split wires, the higher-z TPC of a drift volume is drawn shifted along the wires """
        return self._split_wire and self._drift_volume_of(plane_id)[1] > 0

    def _wire_range(self, plane):
        if self._split_wire:
            return plane.n_wires * 2 + self._cathodeGap
        return plane.n_wires

    def _compute_plane_mix(self):
        """
        For planes of TPC 0 in each cryostat: the parallel-wire plane in each
        other drift volume, at the same z position. These are drawn stacked in
        one view across the cathode.
        """
        mix = {}
        for cryo in self.detector.cryostats:
            volumes = cryo.drift_volumes()
            for plane in cryo.tpcs[0].planes:
                _, z = self._drift_volume_of(plane.id)
                partners = []
                for volume in volumes:
                    tpc = volume[min(z, len(volume) - 1)]
                    if tpc.id.tpc == plane.id.tpc:
                        continue
                    partners += [self._plane_index[p.id] for p in tpc.planes if p.is_parallel_to(plane)]
                mix[self._plane_index[plane.id]] = partners
        return mix

    # --- legacy interface ---

    def name(self):
        return self.detector.name

    def halfwidth(self):
        return self.detector.cryostats[0].half_width

    def halfheight(self):
        return self.detector.cryostats[0].half_height

    def length(self):
        return self.detector.cryostats[0].length

    def nViews(self):
        return len(self._planes)

    def nTPCs(self):
        """ TPCs per cryostat as displayed: split-wire TPC pairs count once """
        cryo = self.detector.cryostats[0]
        return len(cryo.drift_volumes()) if self._split_wire else len(cryo.tpcs)

    def nCryos(self):
        return len(self.detector.cryostats)

    def nPlanes(self):
        return len(self.detector.cryostats[0].tpcs[0].planes)

    def splitWire(self):
        return self._split_wire

    def viewNames(self):
        return self.display.get('view_names', [])

    def planeFlip(self):
        return self._plane_flip

    def planeMix(self):
        return self._plane_mix

    def flipPlane(self, plane_id):
        return self._plane_flip[plane_id]

    def shiftPlane(self, plane_id):
        return self._plane_shift[plane_id]

    def projectionsMatch(self):
        """ True if planes stacked across TPCs always have the same plane number """
        return all(self._planes[partner].id.plane == self._planes[plane].id.plane
                   for plane, partners in self._plane_mix.items() for partner in partners)

    def getPlaneID(self, plane, tpc, cryo):
        if (plane, tpc, cryo) not in self._planeid_map:
            raise LookupError(f'{plane}, {tpc}, {cryo} not available.')
        return self._planeid_map[(plane, tpc, cryo)]

    def getOtherPlanes(self, plane_id):
        """ Planes with parallel wires in the other TPCs of the cryostat, by TPC number """
        return [self._plane_index[p] for p in self.detector.parallel_planes(self._planes[plane_id].id)]

    def tRange(self):
        return self._tRange

    def wRange(self, plane):
        return self._wRange[plane]

    def readoutWindowSize(self):
        return self._readoutWindowSize

    def override_time_range(self, time_range):
        print('!!!WARNING!!!')
        print(f'Overriding time range from {self._tRange} to {time_range}.')
        print('!!!WARNING!!!')
        self._tRange = time_range
        self._readoutWindowSize = time_range

    def readoutPadding(self):
        return self.display.get('readout_padding', 0)

    def time2cm(self):
        return self._time2Cm

    def wire2cm(self):
        return self._planes[0].wire_pitch

    def aspectRatio(self):
        return self.wire2cm() / self.time2cm()

    def samplingRate(self):
        """ TPC tick period in ns """
        return self.detector.timing.tpc_tick_period * 1000.

    def triggerOffset(self):
        return self.detector.timing.trigger_offset_tpc_ticks

    def offset(self, plane):
        if self.display.get('apply_trigger_offset', False):
            return self.triggerOffset() * self.time2cm()
        return 0.

    def timeOffsetTicks(self, plane):
        return self.display.get('time_offset_ticks', 0)

    def cathodeGap(self):
        """ display gap between TPCs stacked across the cathode, in ticks """
        return self._cathodeGap

    def anodeCathodeTicks(self, plane, cryo=0):
        """
        For each TPC stacked along the time axis in the view of this plane
        number: (anode tick, cathode tick, flipped) for charge at t0 = 0, in
        that TPC's own ticks (before flipping and stacking).
        The anode is the collection plane, where tick = trigger offset; the
        cathode is the TPC boundary opposite to it.
        """
        first = self.getPlaneID(plane=plane, tpc=0, cryo=cryo)
        result = []
        for index in [first] + self._plane_mix.get(first, []):
            plane_geo = self._planes[index]
            tpc = self._tpc_of_plane(plane_geo.id)
            anode_x = tpc.planes[-1].center[0]
            cathode_x = tpc.bounds.max[0] if tpc.drift_dir[0] < 0 else tpc.bounds.min[0]
            result.append((float(plane_geo.x_to_ticks(anode_x)), float(plane_geo.x_to_ticks(cathode_x)),
                           self._plane_flip[index]))
        return result

    def getLevels(self, plane):
        return self._levels[plane]

    def getPedestal(self, plane):
        return self._pedestals[plane]

    def colorMap(self, plane, colormaptype='default'):
        return _tuples(copy.deepcopy(self.display['color_schemes'][colormaptype][plane]))

    def hasLogo(self):
        return self.display.get('show_logo', False)

    def logo(self):
        return str(LOGO_PATH / self.display['logo'])

    def logoScale(self):
        return self.display.get('logo_scale', 1.)

    def logoPos(self):
        return self.display.get('logo_pos', [0, 0])

    # --- optical ---

    def opdetLoc(self):
        center = self.detector.opdets.center
        return list(center[:, 0]), list(center[:, 1]), list(center[:, 2])

    def opdetName(self):
        if 'opdet_names' in self.display:
            return self.display['opdet_names']
        return [OPDET_NAME_BY_SHAPE.get(s, 'unknown') for s in self.detector.opdets.shape]

    def opdetRadius(self):
        return self.display.get('opdet_radius', 10.16)

    def opdetToTPC(self, ch):
        """ Index, ordered in x across the detector, of the drift volume the op det faces """
        x = self.detector.opdets.center[ch][0]
        volumes = sorted((v for c in self.detector.cryostats for v in c.drift_volumes()),
                         key=lambda v: v[0].bounds.min[0])
        distance = [max(v[0].bounds.min[0] - x, 0., x - v[0].bounds.max[0]) for v in volumes]
        return int(np.argmin(distance))

    def opdetDefaultValue(self):
        return np.nan

    def nOpticalFrames(self):
        return self.display.get('n_optical_frames', 1)

    def opticalOffset(self):
        return self.display.get('optical_offset', 0)

    # --- CRT (SBND only for now) ---

    def _crt(self, key):
        return self.display.get('crt', {}).get(key)

    @property
    def crt_top_ymin(self):
        return self._crt('top_ymin')

    @property
    def crt_bot_ymax(self):
        return self._crt('bot_ymax')

    @property
    def crt_front_zmin(self):
        return self._crt('front_zmin')

    @property
    def crt_back_zmax(self):
        return self._crt('back_zmax')

    # --- LArSoft service objects, for the C++ drawers ---

    def _service(self, name):
        service = getattr(self.services, name)
        if service is None:
            raise RuntimeError(f'LArSoft service {name!r} is not loaded; this drawer still needs it')
        return service

    def getGeometryCore(self):
        return self._service('geometry_core')

    def getDetectorProperties(self):
        return self._service('detector_properties')

    def getWireReadout(self):
        return self._service('wire_readout')

    def getDetectorClocks(self):
        return self._service('detector_clocks')

    def getLArProperties(self):
        return self._service('lar_properties')

    def getAuxDetGeometryCore(self):
        return self._service('aux_det_geometry')

    def print_summary(self):
        t = self.detector.timing
        print(f'Configured {self.name()} from {self.detector.gdml}:')
        print('Trigger Offset TPC:     ', self.triggerOffset())
        print('Time Range:             ', self._tRange)
        print('Readout Window Size:    ', self._readoutWindowSize)
        print('Trigger Time            ', t.trigger_time)
        print('Tick Period:            ', t.tpc_tick_period)
        print('Drift Velocity:         ', t.drift_velocity)
        print('Time To Cm:             ', self.time2cm())
        print('Wire To Cm:             ', self.wire2cm())
        print('')
