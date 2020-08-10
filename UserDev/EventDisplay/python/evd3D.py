#!/usr/bin/env python
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

try:
    import pyqtgraph.opengl as gl
except:
    print("ERROR: Must have opengl for this viewer")
    exit()

from gui import evdgui3D
import argparse
import sys
import signal
from pyqtgraph.Qt import QtGui, QtCore

from ROOT import evd

from evdmanager import geometry, evd_manager_3D

try:
    import SBNDservices as services
except:
    print ('Did not find SBND services.')
    pass

try:
    import ICARUSservices as services
except:
    print ('Did not find ICARUS services.')
    pass

# This is to allow key commands to work when focus is on a box




def sigintHandler(*args):
    """Handler for the SIGINT signal."""
    sys.stderr.write('\r')
    sys.exit()


def main():

    parser = argparse.ArgumentParser(description='Python based 3D event display.  Requires opengl.')
    geom = parser.add_mutually_exclusive_group()
    geom.add_argument('-A', '-a', '--argoneut',
                      action='store_true',
                      help="Run with the argoneut geometry")
    geom.add_argument('-U', '-u', '--uboone',
                      action='store_true',
                      help="Run with the microboone geometry")
    geom.add_argument('-S', '-s', '--sbnd',
                      action='store_true',
                      help="Run with the SBND Geometry")
    geom.add_argument('-I', '-i', '--icarus',
                      action='store_true',
                      help="Run with the ICARUS Geometry")
    geom.add_argument('-L', '-l', '--lariat',
                      action='store_true',
                      help="Run with the lariat geometry")
    parser.add_argument('file', nargs='*', help="Optional input file to use")

    #
    # Fhicl configuration
    #
    parser.add_argument('--config',
                        dest="config_path",
                        help="Configuration file path (must define `services` or host serviceTable below)")
    parser.add_argument('--table',
                        dest="service_table",
                        help="Name of the FHiCL table where all services are configured")

    args = parser.parse_args()

    if args.config_path is not None:
        services.ServiceManager.setConfiguration(args.config_path, args.service_table)

    app = QtGui.QApplication(sys.argv)

    geometryCore    = services.ServiceManager('Geometry')
    detProperties   = services.ServiceManager('DetectorProperties')
    detClocks       = services.ServiceManager('DetectorClocks')
    lar_properties  = services.ServiceManager('LArProperties')

    if args.uboone:
        geom = geometry.microboone()
    elif args.lariat:
        geom = geometry.lariat()
    elif args.sbnd:
        geom = geometry.sbnd(geometryCore,detProperties,detClocks,lar_properties)
    elif args.icarus:
        geom = geometry.icarus(geometryCore,detProperties,detClocks,lar_properties)
    else:
        geom = geometry.argoneut()

    # If a file was passed, give it to the manager:

    manager = evd_manager_3D(geom)
    manager.setInputFiles(args.file)


    thisgui = evdgui3D(geom, manager)
    # manager.goToEvent(0)

    signal.signal(signal.SIGINT, sigintHandler)
    timer = QtCore.QTimer()
    timer.start(500)  # You may change this if you wish.
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.

    app.exec_()
    # sys.exit(app.exec_())


if __name__ == '__main__':
    main()
