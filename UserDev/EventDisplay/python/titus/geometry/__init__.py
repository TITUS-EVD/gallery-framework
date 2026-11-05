"""
Detector geometry without LArSoft: detector descriptions are loaded from JSON
files are dumped from the LArSoft services (tools/dump_detector.py).
"""

import json
import pathlib

from .detector import (Box, ChannelMap, Cryostat, Detector, OpDets, Plane,
                       PlaneID, Timing, TPC, TPCID)

DETECTOR_DIR = pathlib.Path(__file__).parent.parent / 'detectors'

# variants of a GDML with the same TITUS description: wires aren't in the
# GDML (LArSoft v10 builds them from the wire readout configuration)
_GDML_VARIANT_SUFFIXES = ('_nowires',)


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


def default_gdml(detector, directory=DETECTOR_DIR):
    """ the GDML drawn until a file says which it was made with (<detector>.display.json), or None """
    path = pathlib.Path(directory) / f'{detector}.display.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get('default_gdml')


def find_gdml(detector, gdml):
    """
    The GDML of a shipped description for a GDML name (e.g. 'sbnd_v02_06',
    or its variant 'sbnd_v02_06_nowires'), None if TITUS has none for it
    """
    gdmls = {g for d, g in available_detectors() if d == detector}
    if gdml in gdmls:
        return gdml
    for suffix in _GDML_VARIANT_SUFFIXES:
        if gdml.endswith(suffix) and gdml[:-len(suffix)] in gdmls:
            return gdml[:-len(suffix)]
    return None


def load_detector(detector, gdml=None):
    """
    Load a TITUS detector description by detector name ('sbnd', 'icarus'),
    optionally for a specific GDML (e.g. 'sbnd_v02_04'), or from a path.
    Without a GDML: the detector's only description, else its default_gdml.
    """
    path = pathlib.Path(detector)
    if path.suffix in ('.json', '.gz') and path.exists():
        return Detector.load(path)

    if gdml is not None:
        gdml = find_gdml(detector, gdml) or gdml
    matches = {k: v for k, v in available_detectors().items()
               if k[0] == detector and (gdml is None or k[1] == gdml)}
    if not matches:
        known = ', '.join(f'{d} ({g})' for d, g in available_detectors())
        raise LookupError(f'no detector description for {detector!r}'
                          + (f' with gdml {gdml!r}' if gdml else '') + f'; available: {known}')
    if len(matches) > 1:
        default = default_gdml(detector)
        if (detector, default) not in matches:
            raise LookupError(f'several geometries for {detector!r}, specify one of '
                              f'{sorted(g for _, g in matches)}')
        return Detector.load(matches[(detector, default)])
    return Detector.load(next(iter(matches.values())))


# after load_detector: the geometry module imports it
from .geometry import Geometry, load_display_settings  # noqa: E402
