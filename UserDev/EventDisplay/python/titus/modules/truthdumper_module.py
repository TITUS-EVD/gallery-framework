#!/usr/bin/env python3

"""
Prints event truth information to the terminal.
This module would be a nice place to add print options to the GUI too
"""

from .module import Module
from ..readers import ProductNotFound


class TruthDumperModule(Module):
    def __init__(self, tag='generator'):
        super().__init__()
        self._tpo = TruthPrintOut(tag)

    def update(self):
        self._tpo.print(self._gi.event_handle())


class TruthPrintOut:
    _HEADER = '''++++++++++++++++++++++++++++++++++++++++++++++++++++
{}Truth information for Run {} Subrun{} Event {}
++++++++++++++++++++++++++++++++++++++++++++++++++++'''
    _FOOTER = '++++++++++++++++++++++++++++++++++++++++++++++++++++'
    _INDENT = '  '
    NU_NAME_MAP = {
        12: 'NuE',
        14: 'NuMu',
        16: 'NuTau',
        -12: 'NuEBar',
        -14: 'NuMuBar',
        -16: 'NuTauBar',
    }

    def __init__(self, tag='generator'):
        # simb::MCTruth products of this module label
        self._tag = tag

    def print(self, gallery_event):
        ev_aux = gallery_event.eventAuxiliary()
        print(TruthPrintOut._HEADER.format(TruthPrintOut._INDENT,
            ev_aux.run(), ev_aux.subRun(), ev_aux.event()))
        self._print_info(gallery_event)
        print(TruthPrintOut._FOOTER)

    def _print_info(self, ev):
        from ..readers.gallery import read_event_product, titusio
        try:
            truths = read_event_product(titusio(), ev, 'mctruths', self._tag)
        except ProductNotFound as e:
            print(f'{TruthPrintOut._INDENT}Could not load truth information from this file')
            print(f'{TruthPrintOut._INDENT}{e}')
            return
        for line in self.lines(truths):
            print(f'{TruthPrintOut._INDENT}{line}')

    @staticmethod
    def lines(truths):
        ''' one line per neutrino interaction of an MCTruths container '''
        out = []
        for row in range(len(truths)):
            if not truths.has_neutrino[row]:
                continue
            name = TruthPrintOut.NU_NAME_MAP.get(int(truths.nu_pdg[row]), str(truths.nu_pdg[row]))
            ccnc = 'CC' if truths.ccnc[row] == 0 else 'NC'
            vertex = (float(truths.vx[row]), float(truths.vy[row]), float(truths.vz[row]))
            out.append(f'{name} {ccnc} interaction (mode={truths.interaction_type[row]}) at {vertex}, '
                       f'Enu={truths.nu_energy[row]:.4f} GeV')
        return out
