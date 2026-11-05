"""
The configuration of the jobs that made an art-root file (titus.readers.file_config):
the RootFileDB SQLite database found where TFile says its key is, and the
GDML in its parameter sets. ROOT is faked: a file with a database at an offset.
"""

import sqlite3
from types import SimpleNamespace

from titus.readers.file_config import (FileConfig, config_from_parameter_sets, file_config, file_gdml,
                                       gdml_from_parameter_sets, gdml_name, parameter_sets, root_file_db,
                                       sce_map_from_parameter_sets)

# parameter sets as art stores them (FHiCL, one line), e.g. the Geometry service's
GEOMETRY = 'services:{Geometry:{GDML:"sbnd_v02_06.gdml" Name:"sbndv2" SurfaceY:0}}'
CLOCKS = 'services:{DetectorClocksService:{G4RefTime:-1700 TriggerOffsetTPC:-205}}'
# as in an SBND v10_14 file (the calibration map's name ends in InputFilename too)
SPACE_CHARGE = ('CalibrationInputFilename:"SCEoffsets.root" EnableCalEfieldSCE:true EnableCalSpatialSCE:true '
                'EnableCorrSCE:false EnableSimEfieldSCE:true EnableSimSpatialSCE:true '
                'InputFilename:"SCEoffsets/SCEoffsets_SBND_E500_dualmap_CV_voxelTH3.root" '
                'RepresentationType:"Voxelized_TH3" is2DdriftSimHack:true '
                'service_provider:"SpaceChargeServiceSBND" service_type:"SpaceCharge"')


def database(tmp_path, blobs):
    ''' the bytes of an SQLite database with art's ParameterSets table '''
    path = tmp_path / 'db.sqlite'
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE ParameterSets (ID PRIMARY KEY, PSetBlob)')
    connection.executemany('INSERT INTO ParameterSets VALUES (?, ?)', [(str(i), b) for i, b in enumerate(blobs)])
    connection.commit()
    connection.close()
    return path.read_bytes()


def art_file(tmp_path, db, compressed=False):
    ''' (fake TFile, path) of a file holding the database under a RootFileDB key, as art writes it '''
    path = tmp_path / 'file.root'
    junk, key_header = b'root' + bytes(1000), bytes(64)
    path.write_bytes(junk + key_header + db + bytes(100))
    key = SimpleNamespace(GetSeekKey=lambda: len(junk), GetKeylen=lambda: len(key_header),
                          GetNbytes=lambda: len(key_header) + len(db),
                          GetObjlen=lambda: len(db) * (3 if compressed else 1))
    tfile = SimpleNamespace(GetKey=lambda name: key if name == 'RootFileDB' else None)
    return tfile, str(path)


def test_gdml_name():
    assert gdml_name('sbnd_v02_06.gdml') == 'sbnd_v02_06'
    assert gdml_name('/cvmfs/.../gdml/sbnd_v02_06_nowires.gdml') == 'sbnd_v02_06_nowires'
    assert gdml_name('icarus_refactored_nounderscore_20230918') == 'icarus_refactored_nounderscore_20230918'


def test_gdml_from_parameter_sets():
    assert gdml_from_parameter_sets([CLOCKS, GEOMETRY]) == 'sbnd_v02_06'
    assert gdml_from_parameter_sets([CLOCKS]) is None
    # processes that disagree: the GDML most of them loaded
    old = GEOMETRY.replace('v02_06', 'v02_04')
    assert gdml_from_parameter_sets([old, GEOMETRY, GEOMETRY]) == 'sbnd_v02_06'


def test_the_space_charge_map_the_simulation_used():
    assert sce_map_from_parameter_sets([GEOMETRY, SPACE_CHARGE]) == 'SCEoffsets_SBND_E500_dualmap_CV_voxelTH3'
    # off in simulation (e.g. a reco job's service): not displaced
    assert sce_map_from_parameter_sets([SPACE_CHARGE.replace('EnableSimSpatialSCE:true', 'EnableSimSpatialSCE:false')]) is None
    # a job that simulated it counts, whatever the others loaded
    reco = SPACE_CHARGE.replace('EnableSimSpatialSCE:true', 'EnableSimSpatialSCE:false')
    assert sce_map_from_parameter_sets([reco, SPACE_CHARGE]) == 'SCEoffsets_SBND_E500_dualmap_CV_voxelTH3'
    assert sce_map_from_parameter_sets([GEOMETRY, CLOCKS]) is None
    assert config_from_parameter_sets([GEOMETRY, SPACE_CHARGE]) == \
        FileConfig('sbnd_v02_06', 'SCEoffsets_SBND_E500_dualmap_CV_voxelTH3')


def test_the_configuration_of_a_file(tmp_path):
    tfile, path = art_file(tmp_path, database(tmp_path, [CLOCKS, GEOMETRY, SPACE_CHARGE]))
    assert parameter_sets(root_file_db(tfile, path)) == [CLOCKS, GEOMETRY, SPACE_CHARGE]
    assert file_gdml(tfile, path) == 'sbnd_v02_06'
    assert file_config(tfile, path) == FileConfig('sbnd_v02_06', 'SCEoffsets_SBND_E500_dualmap_CV_voxelTH3')
    # unreadable: nothing known
    assert file_config(tfile, 'root://host//file.root') == FileConfig()


def test_files_without_a_readable_configuration(tmp_path):
    db = database(tmp_path, [GEOMETRY])
    # no RootFileDB key, e.g. TITUS's test fakes
    tfile, path = art_file(tmp_path, db)
    assert file_gdml(SimpleNamespace(GetKey=lambda name: None), path) is None
    assert file_gdml(SimpleNamespace(), path) is None
    # compressed: not a database we know
    assert file_gdml(art_file(tmp_path, db, compressed=True)[0], path) is None
    # remote files aren't read
    assert file_gdml(tfile, 'root://host//file.root') is None
    # not a database there
    tfile, path = art_file(tmp_path, b'not a database' + bytes(100))
    assert file_gdml(tfile, path) is None
