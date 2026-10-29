#!/usr/bin/env python
import sys
import argparse
import signal

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
# No PyROOT thread polling ROOT's event loop (gSystem.ProcessEvents): TITUS
# draws with Qt, and that thread running ROOT concurrently with the file
# opening thread crashed it. Must be set before ROOT initializes (gROOT below).
ROOT.PyConfig.StartGUIThread = False
ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kError;")

from PyQt5 import QtCore, QtWidgets

from titus.gui import Gui
from titus.modules import LArSoftModule, RunModule, GeometryModule, ViewSelectModule, \
    TpcModule, OpDetModule, CrtModule, HelpModule, TruthDumperModule
from titus.gallery_interface import GalleryInterface
from titus.gallery_interface.gallery_interface import threads_enabled


def sigintHandler(*args):
    sys.stderr.write('\r')
    sys.exit()


def main():
    parser = argparse.ArgumentParser(description='TITUS event display.')
    geom_args = parser.add_mutually_exclusive_group()
    geom_args.add_argument('-S', '-s', '--sbnd',
                      action='store_true',
                      help="Run with the SBND Geometry")
    geom_args.add_argument('-I', '-i', '--icarus',
                      action='store_true',
                      help="Run with the ICARUS Geometry")
    parser.add_argument('--timerange',
                      type=int,
                      help="Overrides the number of time samples.")
    parser.add_argument('--threads', action='store_true',
                      help="Experimental: open files in a background thread (crashed ROOT 6.28 + gallery)")
    parser.add_argument('--no-threads', action='store_true',
                      help="Open files on the GUI thread (the default)")
    parser.add_argument('file', nargs='*', help="Optional input file to use")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    gi = GalleryInterface(threaded=(args.threads or threads_enabled()) and not args.no_threads)

    gui = Gui(gi)

    # add basic run controls
    gui.add_module(RunModule())

    # these two modules are required to initialize the view modules
    det_service_name = 'sbnd'
    if args.icarus:
        det_service_name = 'icarus'
    # add other services here...

    lsm = LArSoftModule(det_service_name, args.timerange)
    # files from the command line are opened once the window and modules are
    # set up, in the background, as from the file dialog
    if args.file:
        # (after the geometry and views, which also wait for ready)
        lsm.ready.connect(lambda: QtCore.QTimer.singleShot(0, lambda: gi.open_files(args.file)))
    gm = GeometryModule(lsm)
    

    # optional to add it to the gui. Allows the user to select different
    # geometries during runtime, but if there is only one available it is loaded
    # by default
    # gui.add_module(gm)
    # call this method when using geometry module but not adding it to the gui (no geometry menu added)
    gm.add_gallery_interface(gi)

    # views: TPC, Optical, CRT
    vm = ViewSelectModule()
    gui.add_module(vm)

    vm.add_module(TpcModule(lsm, gm), view_name='TPC')
    vm.add_module(OpDetModule(lsm, gm), view_name='Optical')
    if args.sbnd:
        vm.add_module(CrtModule(lsm, gm), view_name='CRT')

    # TODO
    # add other modules here. LArSoft module initiates a loading sequence which
    # overwrites the main GUI window, so we put it after view select module.
    # there should be a more elegant way to stage this
    gui.add_module(lsm)
    gui.add_module(HelpModule())
    # gui.add_module(TruthDumperModule())

    # allow no-questions-asked keyboard interrupt 
    signal.signal(signal.SIGINT, sigintHandler)

    # Let the interpreter run each 250 ms.
    timer = QtCore.QTimer()
    timer.start(250)
    timer.timeout.connect(lambda: None)  

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
