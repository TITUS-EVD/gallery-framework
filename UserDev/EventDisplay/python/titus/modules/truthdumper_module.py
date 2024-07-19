#!/usr/bin/env python3

"""
Prints event truth information to the terminal.
This module would be a nice place to add print options to the GUI too
"""

import ROOT

from .module import Module


import galleryUtils
sc = galleryUtils.SourceCode
sc.loadHeaderFromUPS('nusimdata/SimulationBase/MCTruth.h')


class TruthDumperModule(Module):
    def __init__(self):
        super().__init__()
        self._tpo = TruthPrintOut()

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

    def __init__(self):
        galleryUtils.make_getValidHandle.make('std::vector<simb::MCTruth>')

    def print(self, gallery_event):
        ev_aux = gallery_event.eventAuxiliary()
        print(TruthPrintOut._HEADER.format(TruthPrintOut._INDENT,
            ev_aux.run(), ev_aux.subRun(), ev_aux.event()))
        self._print_info(gallery_event)
        print(TruthPrintOut._FOOTER)

    def _print_info(self, ev):
        try:
            get_mctruths = ev.getValidHandle[ROOT.vector(ROOT.simb.MCTruth)]
            mctruths_tag = ROOT.art.InputTag("generator");
            truth_vec = get_mctruths(mctruths_tag)
        except TypeError as e:
            print(f'{TruthPrintOut._INDENT}Could not load truth information from this file')
            print(f'{TruthPrintOut._INDENT}{e}')
            return

        for idx in range(truth_vec.size()):
            t = truth_vec.at(idx)
            nutype = t.GetNeutrino().Nu().PdgCode()
            ccnc = "CC" if not t.GetNeutrino().CCNC() else "NC"
            origin = t.Origin()
            vertex = (t.GetNeutrino().Nu().Vx(), t.GetNeutrino().Nu().Vy(), t.GetNeutrino().Nu().Vz())
            enu = t.GetNeutrino().Nu().E()
            mode = t.GetNeutrino().InteractionType()

            print(f"{TruthPrintOut._INDENT}{TruthPrintOut.NU_NAME_MAP[nutype]} {ccnc} interaction (mode={mode}) at {vertex}, Enu={enu:.4f} GeV")
