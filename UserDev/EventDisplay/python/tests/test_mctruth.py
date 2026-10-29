"""
The MC truth dock's text: particle names and masses, GENIE status codes, the
interaction summary and the particle table (titus.display.mctruth).
"""

import pathlib

import numpy as np
import pytest

from titus.display.mctruth import (particle_mass, particle_name, particle_rows, status_name, truth_label,
                                   truth_summary)
from titus.readers import DumpReader

STEP3 = pathlib.Path(__file__).parent / 'data' / 'sbnd-reco2-step3'


@pytest.fixture(scope='module')
def genie():
    return DumpReader(STEP3).get(0, 'mctruths', 'generator')


@pytest.fixture(scope='module')
def corsika():
    return DumpReader(STEP3).get(0, 'mctruths', 'corsika')


def test_particle_names_and_masses():
    assert [particle_name(p) for p in (13, -13, 2212, 111, 14, -12)] == ['μ⁻', 'μ⁺', 'p', 'π⁰', 'νμ', 'ν̄e']
    # nuclei (10LZZZAAAI) and GENIE's own codes
    assert particle_name(1000180400) == 'Ar40' and particle_name(1000170390) == 'Cl39'
    assert particle_name(2000000300) == 'GENIE 2000000300' and particle_name(9999) == '9999'
    assert particle_mass(2212) == pytest.approx(0.938272) and particle_mass(22) == 0.
    assert particle_mass(1000180400) == pytest.approx(40 * 0.931494) and particle_mass(2000000300) is None
    assert status_name(1) == 'final state' and status_name(14) == 'hadron in nucleus' and status_name(99) == 'status 99'


def test_genie_final_state(genie):
    rows = particle_rows(genie, 0)
    pdg, status, energy = genie.particles(0)
    assert len(rows) == (status == 1).sum() and {r.status for r in rows} == {'final state'}
    proton = next(r for r in rows if r.pdg == 2212)
    assert proton.kinetic_energy == pytest.approx((proton.energy - 0.938272) * 1000, abs=1e-3)
    assert all(r.kinetic_energy >= 0 for r in rows)
    # every particle, including the initial state and the nucleus
    everything = particle_rows(genie, 0, final_state_only=False)
    assert len(everything) == len(pdg)
    assert [r.name for r in everything[:2]] == ['νμ', 'Ar40'] and everything[1].status == 'initial state'


def test_genie_summary(genie):
    lines = dict(truth_summary(genie, 0, tpc_of=lambda xyz: '0', t0_us=0.5))
    assert lines['Neutrino'] == 'νμ (14), 1.526 GeV'
    assert lines['Interaction'] == 'NC RES (type 1006)'
    assert lines['Target'] == 'Ar40'
    assert lines['Vertex'].endswith('cm, TPC 0') and lines['Time'] == '0.50 μs from the trigger'
    assert lines['Final state'].startswith('11 particles: 6 p, 4 n, 1 νμ')
    assert truth_label(genie, 0, 'generator') == 'νμ NC (generator 1)'
    outside = dict(truth_summary(genie, 0, tpc_of=lambda xyz: None))
    assert outside['Vertex'].endswith('cm, outside the TPCs') and 'Time' not in outside


def test_cosmic_summary(corsika):
    lines = dict(truth_summary(corsika, 0))
    assert lines['Origin'] == 'Cosmic' and 'Neutrino' not in lines
    assert lines['Final state'].startswith('170 particles: 54 μ⁺, 53 μ⁻, 37 n')
    _, _, energy = corsika.particles(0)
    assert lines['Energies'] == f'{energy.min():.3g} to {energy.max():.3g} GeV'
    assert truth_label(corsika, 0, 'corsika') == 'Cosmic (corsika 1)'
    assert len(particle_rows(corsika, 0)) == 170 and np.isfinite([r.energy for r in particle_rows(corsika, 0)]).all()
