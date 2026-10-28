import json
import os
import pathlib

import pytest

from titus.geometry import available_detectors, load_detector

DATA_DIR = pathlib.Path(__file__).parent / 'data'

# drawable tests run without a display (gpvm, CI)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# (detector, gdml) of every shipped description that has a golden table
GOLDEN = [key for key in available_detectors()
          if (DATA_DIR / f'{key[0]}-{key[1]}.golden.json').exists()]


@pytest.fixture(scope='session', params=GOLDEN, ids=lambda k: f'{k[0]}-{k[1]}')
def detector_and_golden(request):
    detector, gdml = request.param
    with open(DATA_DIR / f'{detector}-{gdml}.golden.json') as f:
        golden = json.load(f)
    return load_detector(detector, gdml), golden


@pytest.fixture(scope='session')
def sbnd():
    return load_detector('sbnd')


@pytest.fixture(scope='session')
def icarus():
    return load_detector('icarus')
