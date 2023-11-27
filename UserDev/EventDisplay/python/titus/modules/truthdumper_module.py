#!/usr/bin/env python3

"""
Prints event truth information to the terminal.
This module would be a nice place to add print options to the GUI too
"""

import ROOT

from .module import Module


def read_header(h):
    ROOT.gROOT.ProcessLine('#include "%s"' % h)

def provide_get_valid_handle(class_):
    ROOT.gROOT.ProcessLine('template gallery::ValidHandle<%(name)s> gallery::Event::getValidHandle<%(name)s>(art::InputTag const&) const;' % {'name' : class_})


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

    def __init__(self):
        read_header('gallery/ValidHandle.h')
        provide_get_valid_handle('std::vector<simb::MCTruth>')

    def print(self, gallery_event):
        ev_aux = gallery_event.eventAuxiliary()
        print(TruthPrintOut._HEADER.format(TruthPrintOut._INDENT,
            ev_aux.run(), ev_aux.subRun(), ev_aux.event()))
        self._print_info(gallery_event)
        print(TruthPrintOut._FOOTER)

    def _print_info(self, ev):
        try:
            get_mctruths = ev.getValidHandle(ROOT.vector(ROOT.simb.MCTruth))
            mctruths_tag = ROOT.art.InputTag("generator");
            truth = get_mctruths(mctruths_tag)
        except TypeError as e:
            print(f'{TruthPrintOut._INDENT}Could not load truth information from this file')
            print(f'{TruthPrintOut._INDENT}{e}')
            return

        origin = truth.Origin()
        vertex = (truth.GetNeutrino().Nu().Vx(), truth.GetNeutrino().Nu().Vy(), truth.GetNeutrino().Nu().Vz())
        nutype = truth.GetNeutrino().Nu().PdgCode()
        enu = truth.GetNeutrino().Nu().E()
        print(origin, vertex, nutype, enu)

