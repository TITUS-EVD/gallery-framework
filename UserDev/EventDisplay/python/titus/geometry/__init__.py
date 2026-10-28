"""
Detector geometry without LArSoft: detector descriptions are loaded from JSON
files are dumped from the LArSoft services (tools/dump_detector.py).
"""

import pathlib

from .detector import (Box, ChannelMap, Cryostat, Detector, OpDets, Plane,
                       PlaneID, Timing, TPC, TPCID)

DETECTOR_DIR = pathlib.Path(__file__).parent.parent / 'detectors'


def available_detectors():
    """ Detector description files included with TITUS, as {(detector, gdml): path} """
    found = {}
    for path in sorted(DETECTOR_DIR.glob('*.json*')):
        name = path.name.split('.json')[0]
        if '.' in name:
            # e.g. display configuration files: <detector>.display.json
            continue
        detector, _, gdml = name.partition('-')
        found[(detector, gdml)] = path
    return found


def load_detector(detector, gdml=None):
    """
    Load a TITUS detector description by detector name ('sbnd', 'icarus'),
    optionally for a specific GDML (e.g. 'sbnd_v02_04'), or from a path.
    """
    path = pathlib.Path(detector)
    if path.suffix in ('.json', '.gz') and path.exists():
        return Detector.load(path)

    matches = {k: v for k, v in available_detectors().items()
               if k[0] == detector and (gdml is None or k[1] == gdml)}
    if not matches:
        known = ', '.join(f'{d} ({g})' for d, g in available_detectors())
        raise LookupError(f'no detector description for {detector!r}'
                          + (f' with gdml {gdml!r}' if gdml else '') + f'; available: {known}')
    if len(matches) > 1:
        raise LookupError(f'several geometries for {detector!r}, specify one of '
                          f'{sorted(g for _, g in matches)}')
    return Detector.load(next(iter(matches.values())))


# after load_detector: the geometry module imports it
from .geometry import Geometry, ServiceHandles, load_display_settings  # noqa: E402
