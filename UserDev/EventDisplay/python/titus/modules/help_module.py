#!/usr/bin/env python3

"""
Add the help menu & page(s)
"""

from PyQt5 import QtWidgets, QtGui, QtCore

from titus.modules import Module


TITLE = '<h2 style="text-align:center">TITUS</h2>'
TAGLINE = '<p style="text-align:center">The event display for SBN at Fermilab</p>'

# TODO make this from the package information
VERSION = '<p style="text-align:center">Version 1.3.0</p>'

class HelpModule(Module):
    def __init__(self):
        super().__init__()

    def _initialize(self):
        """ Populate menu bar """
        # add some file controls in their expected places
        help_menu = QtWidgets.QMenu("&Help", self._gui)
        about_action = QtWidgets.QAction('&About', help_menu)
        about_action.triggered.connect(self._about)
        help_menu.addAction(about_action)

        self._gui.menuBar().addMenu(help_menu)

    def _about(self):
        QtWidgets.QMessageBox.about(self._gui, 'About', f'{TITLE}{TAGLINE}{VERSION}')
