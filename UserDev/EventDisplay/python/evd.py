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

# This is to allow key commands to work when focus is on a box




def sigintHandler(*args):
    """Handler for the SIGINT signal."""
    sys.stderr.write('\r')
    sys.exit()


def main():

    parser = argparse.ArgumentParser(description='Python based event display.')
    geom = parser.add_mutually_exclusive_group()
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
    geom.add_argument('-L', '-l', '--lariat',
                      action='store_true',
                      help="Run with the lariat geometry")
    parser.add_argument('file', nargs='*', help="Optional input file to use")

    args = parser.parse_args()

    app = QtGui.QApplication(sys.argv)

    # dark_palette = QtGui.QPalette()    
    # dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    # dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    # dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    # dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    # dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    # dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    # dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    # dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    # dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    # dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    # dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    # dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    # dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)    
    # app.setPalette(dark_palette)    
    # app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

    if args.uboone:
        geom = geometry.microboone()
    elif args.uboonetruncated:
        geom = geometry.microboonetruncated()
    elif args.lariat:
        geom = geometry.lariat()
    elif args.sbnd:
        geom = geometry.sbnd()
    else:
        geom = geometry.argoneut()

    # If a file was passed, give it to the manager:

    manager = evd_manager_2D(geom)
    manager.setInputFiles(args.file)


    thisgui = evdgui(geom, manager)
    # manager.goToEvent(0)

    signal.signal(signal.SIGINT, sigintHandler)
    timer = QtCore.QTimer()
    timer.start(500)  # You may change this if you wish.
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.

    app.exec_()
    # sys.exit(app.exec_())


if __name__ == '__main__':
    main()
