#!/usr/bin/env python
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kError;")


from gui import evdgui
import argparse
import sys
import signal
from pyqtgraph.Qt import QtGui, QtCore


from ROOT import evd

from evdmanager import geometry, evd_manager_2D

try:
    import SBNDservices as services
except:
    pass

try:
    import ICARUSservices as services
except:
    pass


def sigintHandler(*args):
    """Handler for the SIGINT signal."""
    sys.stderr.write('\r')
    sys.exit()


def main():

    parser = argparse.ArgumentParser(description='TITUS event display.')
    geom = parser.add_mutually_exclusive_group()

    #
    # Detector
    #
    geom.add_argument('-A', '-a', '--argoneut',
                      action='store_true',
                      help="Run with the argoneut geometry")
    geom.add_argument('-U', '-u', '--uboone',
                      action='store_true',
                      help="Run with the microboone geometry")
    geom.add_argument('-T', '-t', '--uboonetruncated',
                      action='store_true',
                      help="Run with the truncated microboone geometry")
    geom.add_argument('-S', '-s', '--sbnd',
                      action='store_true',
                      help="Run with the SBND Geometry")
    geom.add_argument('-S3', '-s3', '--sbnd3',
                      action='store_true',
                      help="Run with the SBND Geometry with 3 drift windows")
    geom.add_argument('-S1', '-s1', '--sbnd1',
                      action='store_true',
                      help="Run with the SBND Geometry with 1 drift window")
    geom.add_argument('-I', '-i', '--icarus',
                      action='store_true',
                      help="Run with the ICARUS Geometry")
    geom.add_argument('-L', '-l', '--lariat',
                      action='store_true',
                      help="Run with the lariat geometry")

    #
    # Fhicl configuration
    #
    parser.add_argument('--config',
                        dest="config_path",
                        help="Configuration file path (must define `services` or host serviceTable below)")
    parser.add_argument('--table',
                        dest="service_table",
                        help="Name of the FHiCL table where all services are configured")
    parser.add_argument('-nosw', '--no_split_wire',
                        action='store_true',
                        help="Do not use split wire geometry (you also need to pass the right FHiCL)")

    #
    # Input file
    #
    parser.add_argument('file', nargs='*', help="Optional input file to use")

    args = parser.parse_args()

    if args.config_path is not None:
        services.ServiceManager.setConfiguration(args.config_path, args.service_table)

    detClocks       = services.ServiceManager('DetectorClocks')
    detProperties   = services.ServiceManager('DetectorProperties')
    geometryCore    = services.ServiceManager('Geometry')
    lar_properties  = services.ServiceManager('LArProperties')

    detPropData = detProperties.DataFor(detClocks.DataForJob())

    print("NumberTimeSamples:",detPropData.NumberTimeSamples())

    app = QtGui.QApplication(sys.argv)

    if args.uboone:
        geom = geometry.microboone()
    elif args.uboonetruncated:
        geom = geometry.microboonetruncated()
    elif args.lariat:
        geom = geometry.lariat()
    elif args.sbnd:
        geom = geometry.sbnd(geometryCore,detProperties,detClocks,lar_properties)
    elif args.sbnd3:
        geom = geometry.sbnd(geometryCore,detProperties,detClocks,lar_properties)
        geom._tRange = 7500
        geom._triggerOffset = 2500
        geom._readoutWindowSize = 7500
        geom.recalculateOffsets()
    elif args.sbnd1:
        geom = geometry.sbnd(geometryCore,detProperties,detClocks,lar_properties)
        geom._tRange = 3000
        geom._triggerOffset = 0
        geom._readoutWindowSize = 3000
    elif args.icarus:
        geom = geometry.icarus(geometryCore,detProperties,detClocks,lar_properties,args.no_split_wire)
    else:
        geom = geometry.argoneut()

    # If a file was passed, give it to the manager:

    manager = evd_manager_2D(geom)
    manager.setInputFiles(args.file)


    thisgui = evdgui(geom, manager, app)
    # manager.goToEvent(0)

    signal.signal(signal.SIGINT, sigintHandler)
    timer = QtCore.QTimer()
    timer.start(500)  # You may change this if you wish.
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.

    app.exec_()
    # sys.exit(app.exec_())


if __name__ == '__main__':
    main()
