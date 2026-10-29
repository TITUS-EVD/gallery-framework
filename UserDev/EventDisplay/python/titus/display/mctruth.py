"""
MC truth for the MC truth dock: particle names, masses and GENIE status
codes, an interaction's summary and its particle table. No Qt.
"""

from collections import Counter
from typing import List, NamedTuple, Optional

# simb::Origin_t
ORIGIN_NAMES = {0: 'Unknown', 1: 'Neutrino', 2: 'Cosmic', 3: 'Supernova', 4: 'Single particle'}

# simb::MCNeutrino::Mode()
MODE_NAMES = {0: 'QE', 1: 'RES', 2: 'DIS', 3: 'COH', 4: 'COH elastic', 5: 'electron scattering',
              6: 'IMD annihilation', 7: 'inverse beta decay', 8: 'Glashow resonance', 9: 'AMNuGamma',
              10: 'MEC', 11: 'diffractive', 12: 'EM', 13: 'weak mix'}

# GENIE particle status codes (MCParticle::StatusCode)
STATUS_NAMES = {0: 'initial state', 1: 'final state', 2: 'intermediate', 3: 'decayed',
                11: 'target nucleon', 12: 'pre-fragmentation', 13: 'pre-decay resonance',
                14: 'hadron in nucleus', 15: 'nuclear remnant', 16: 'nucleon cluster'}
FINAL_STATE = 1

# name, mass (GeV)
PARTICLES = {
    11: ('e⁻', 0.000511), -11: ('e⁺', 0.000511), 13: ('μ⁻', 0.105658), -13: ('μ⁺', 0.105658),
    15: ('τ⁻', 1.77686), -15: ('τ⁺', 1.77686),
    12: ('νe', 0.), -12: ('ν̄e', 0.), 14: ('νμ', 0.), -14: ('ν̄μ', 0.), 16: ('ντ', 0.), -16: ('ν̄τ', 0.),
    22: ('γ', 0.), 111: ('π⁰', 0.134977), 211: ('π⁺', 0.139570), -211: ('π⁻', 0.139570),
    221: ('η', 0.547862), 130: ('K⁰L', 0.497611), 310: ('K⁰S', 0.497611), 311: ('K⁰', 0.497611),
    -311: ('K̄⁰', 0.497611), 321: ('K⁺', 0.493677), -321: ('K⁻', 0.493677),
    2212: ('p', 0.938272), -2212: ('p̄', 0.938272), 2112: ('n', 0.939565), -2112: ('n̄', 0.939565),
    3122: ('Λ', 1.115683), 3222: ('Σ⁺', 1.18937), 3212: ('Σ⁰', 1.192642), 3112: ('Σ⁻', 1.197449),
    2224: ('Δ⁺⁺', 1.232), 2214: ('Δ⁺', 1.232), 2114: ('Δ⁰', 1.232), 1114: ('Δ⁻', 1.232),
}
ELEMENTS = ['n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
            'Cl', 'Ar', 'K', 'Ca']
ATOMIC_MASS_UNIT = 0.931494  # GeV


def _nucleus(pdg):
    ''' (Z, A) of a nucleus PDG code 10LZZZAAAI, or None '''
    if 1000000000 <= abs(pdg) < 2000000000:
        return (abs(pdg) // 10000) % 1000, (abs(pdg) // 10) % 1000
    return None


def particle_name(pdg):
    pdg = int(pdg)
    if pdg in PARTICLES:
        return PARTICLES[pdg][0]
    nucleus = _nucleus(pdg)
    if nucleus is not None:
        z, a = nucleus
        return f'{ELEMENTS[z] if z < len(ELEMENTS) else f"Z={z} "}{a}'
    if is_genie_pseudo_particle(pdg):
        return f'GENIE {pdg}'
    return str(pdg)


def is_genie_pseudo_particle(pdg):
    ''' GENIE's own codes (2000000xxx) for e.g. nucleon clusters and the hadronic system '''
    return 2000000000 <= abs(int(pdg)) < 3000000000


def particle_mass(pdg) -> Optional[float]:
    ''' mass in GeV, approximate for nuclei (A atomic mass units); None if unknown '''
    pdg = int(pdg)
    if pdg in PARTICLES:
        return PARTICLES[pdg][1]
    nucleus = _nucleus(pdg)
    if nucleus is not None:
        return nucleus[1] * ATOMIC_MASS_UNIT
    return None


def status_name(status):
    return STATUS_NAMES.get(int(status), f'status {int(status)}')


class ParticleRow(NamedTuple):
    name: str
    pdg: int
    kinetic_energy: Optional[float]  # MeV; None without a known mass
    energy: float  # GeV
    status: str


def particle_rows(truths, row, final_state_only=True) -> List[ParticleRow]:
    ''' the particles of an interaction, as the dock's table lists them '''
    pdg, status, energy = truths.particles(row)
    rows = []
    for p, s, e in zip(pdg, status, energy):
        if final_state_only and s != FINAL_STATE:
            continue
        mass = particle_mass(p)
        kinetic = None if mass is None else max(float(e) - mass, 0.) * 1000.
        rows.append(ParticleRow(particle_name(p), int(p), kinetic, float(e), status_name(s)))
    return rows


def truth_label(truths, row, producer):
    ''' the dropdown entry of an interaction '''
    origin = ORIGIN_NAMES.get(int(truths.origin[row]), f'origin {int(truths.origin[row])}')
    if truths.has_neutrino[row]:
        origin = f'{particle_name(truths.nu_pdg[row])} {"CC" if truths.ccnc[row] == 0 else "NC"}'
    return f'{origin} ({producer} {row + 1})'


def truth_summary(truths, row, tpc_of=None, t0_us=None):
    '''
    (label, value) lines describing an interaction. tpc_of(xyz): the
    TPC of the vertex, or None outside; t0_us: its time from the trigger.
    '''
    pdg, status, energy = truths.particles(row)
    final = status == FINAL_STATE
    lines = []
    origin = int(truths.origin[row])
    if truths.has_neutrino[row]:
        nu = int(truths.nu_pdg[row])
        lines.append(('Neutrino', f'{particle_name(nu)} ({nu}), {truths.nu_energy[row]:.3f} GeV'))
        current = {0: 'CC', 1: 'NC'}.get(int(truths.ccnc[row]), '?')
        mode = MODE_NAMES.get(int(truths.mode[row]), f'mode {int(truths.mode[row])}')
        lines.append(('Interaction', f'{current} {mode} (type {int(truths.interaction_type[row])})'))
        targets = [particle_name(p) for p, s in zip(pdg, status) if s == 0 and _nucleus(p) is not None]
        if targets:
            lines.append(('Target', ', '.join(targets)))
        xyz = (float(truths.vx[row]), float(truths.vy[row]), float(truths.vz[row]))
        where = ''
        if tpc_of is not None:
            tpc = tpc_of(xyz)
            where = ', outside the TPCs' if tpc is None else f', TPC {tpc}'
        lines.append(('Vertex', f'({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) cm{where}'))
        if t0_us is not None:
            lines.append(('Time', f'{t0_us:.2f} μs from the trigger'))
    else:
        lines.append(('Origin', ORIGIN_NAMES.get(origin, f'origin {origin}')))
    counts = Counter(particle_name(p) for p in pdg[final])
    lines.append(('Final state', f'{int(final.sum())} particles: ' +
                  ', '.join(f'{n} {name}' for name, n in counts.most_common(6)) +
                  (', ...' if len(counts) > 6 else '')))
    if not truths.has_neutrino[row] and final.any():
        lines.append(('Energies', f'{energy[final].min():.3g} to {energy[final].max():.3g} GeV'))
    return lines
