"""
The configuration of the jobs that made an art-root file, read without art:
art keeps every process's FHiCL parameter sets, services included, in an
SQLite database stored uncompressed under the file's RootFileDB key. Its
bytes are read where ROOT's TFile says they are, and queried with sqlite3.

TITUS uses it to draw a file with the geometry it was made with (the GDML
the Geometry service loaded), as the GDML changes between productions, e.g.
SBND v10_06 (sbnd_v02_04) and v10_14 (sbnd_v02_06, all 4.2 cm lower in z),
and to know whether its simulation moved the charge by space charge (and
with which map), to draw simulated particles where their charge is.
"""

import collections
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import Optional

ROOT_FILE_DB = 'RootFileDB'
SQLITE_HEADER = b'SQLite format 3\x00'

# the Geometry service's GDML file name in a parameter set blob, e.g. GDML:"sbnd_v02_06.gdml"
_GDML = re.compile(r'\bGDML:"([^"]+)"')
# the SpaceCharge service: whether the simulation displaced the charge, and the map it used
_SCE_SERVICE = re.compile(r'\bservice_type:"SpaceCharge"')
_SIM_SPATIAL_SCE = re.compile(r'\bEnableSimSpatialSCE:true\b')
_SCE_MAP = re.compile(r'\bInputFilename:"([^"]+)"')


@dataclass(frozen=True)
class FileConfig:
    ''' what TITUS uses of the configuration a file was made with; None: not known '''
    # e.g. 'sbnd_v02_06'
    gdml: Optional[str] = None
    # the space charge map the simulation displaced the charge with, e.g.
    # 'SCEoffsets_SBND_E500_dualmap_CV_voxelTH3'; None if it didn't (or not known)
    sce_map: Optional[str] = None


def root_file_db(tfile, path):
    """
    The bytes of the file's RootFileDB (an SQLite database), or None if it
    has none or they can't be read. tfile: the file opened with ROOT's TFile,
    which says where the key is; path: the local file, read there. Remote
    files (root://) aren't read: TFile.ReadBuffer into a Python buffer
    crashes PyROOT.
    """
    if not os.path.isfile(path):
        return None
    try:
        key = tfile.GetKey(ROOT_FILE_DB)
        if not key:
            return None
        start = key.GetSeekKey() + key.GetKeylen()
        size = key.GetNbytes() - key.GetKeylen()
        if size != key.GetObjlen():
            # compressed: art stores it as is, so this isn't a database we know
            return None
        with open(path, 'rb') as f:
            f.seek(start)
            data = f.read(size)
    except Exception:
        return None
    return data if data.startswith(SQLITE_HEADER) else None


def parameter_sets(db_bytes):
    """ the FHiCL parameter set blobs (strings) of a RootFileDB """
    # sqlite3 opens files (Connection.deserialize needs Python 3.11)
    with tempfile.NamedTemporaryFile(suffix='.db') as f:
        f.write(db_bytes)
        f.flush()
        connection = sqlite3.connect(f.name)
        try:
            rows = connection.execute('SELECT PSetBlob FROM ParameterSets').fetchall()
        finally:
            connection.close()
    return [blob.decode() if isinstance(blob, bytes) else str(blob) for (blob,) in rows]


def gdml_name(name):
    """ 'path/to/sbnd_v02_06.gdml' -> 'sbnd_v02_06', the name detector descriptions are keyed by """
    base = os.path.basename(name)
    return base[:-len('.gdml')] if base.endswith('.gdml') else base


def gdml_from_parameter_sets(blobs):
    """
    The GDML (as gdml_name) the file's jobs configured, or None. Processes of
    one file normally agree; if not, the one most of them used.
    """
    found = collections.Counter(gdml_name(m) for blob in blobs for m in _GDML.findall(blob))
    if not found:
        return None
    return found.most_common(1)[0][0]


def map_name(name):
    """ 'SCEoffsets/SCEoffsets_SBND_E500_dualmap_CV_voxelTH3.root' -> 'SCEoffsets_SBND_E500_dualmap_CV_voxelTH3' """
    return os.path.splitext(os.path.basename(name))[0]


def sce_map_from_parameter_sets(blobs):
    """
    The space charge map (as map_name) the file's simulation displaced the
    charge with: from a SpaceCharge service with EnableSimSpatialSCE; None if
    none was. Processes that don't simulate (e.g. reco) may load the service
    with it off: any process that simulated it counts.
    """
    found = collections.Counter(map_name(m) for blob in blobs
                                if _SCE_SERVICE.search(blob) and _SIM_SPATIAL_SCE.search(blob)
                                for m in _SCE_MAP.findall(blob)[:1])
    if not found:
        return None
    return found.most_common(1)[0][0]


def config_from_parameter_sets(blobs):
    return FileConfig(gdml=gdml_from_parameter_sets(blobs), sce_map=sce_map_from_parameter_sets(blobs))


def file_config(tfile, path):
    """ the FileConfig of a file (all None if its configuration can't be read) """
    data = root_file_db(tfile, path)
    if data is None:
        return FileConfig()
    try:
        return config_from_parameter_sets(parameter_sets(data))
    except sqlite3.Error:
        return FileConfig()


def file_gdml(tfile, path):
    """ the GDML a file was made with (e.g. 'sbnd_v02_06'), None if its configuration can't be read """
    return file_config(tfile, path).gdml
