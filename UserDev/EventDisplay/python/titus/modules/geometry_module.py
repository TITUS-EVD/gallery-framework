#!/usr/bin/env python3

"""
This module adds UI elements for selecting different geometries. It requires
the LArSoft services from the LArSoft module
"""

import time
from PyQt5 import QtWidgets, QtGui, QtCore

from .module import Module
from ..gallery_interface import geometry as geometry


class GeometryModule(Module):
    geometryChanged = QtCore.pyqtSignal()

    def __init__(self, larsoft_module):
        super().__init__()

        self._lsm = larsoft_module
        self._lsm.servicesLoaded.connect(self._on_larsoft_services_loaded)

        self._available_geoms = None
        self._geom_menu = None
        self._detector_label = QtWidgets.QLabel()
        self._detector_label.setText('Detector: None')
        self._geom_action_map = {}

        self._time_range = self._lsm.time_range

        self.current_geom = None

    def _initialize(self):
        """ Add list of geometries as controls to the GUI in the menu bar """
        self.populate_menu()
        self._gui.statusBar().addPermanentWidget(self._detector_label)

    def populate_menu(self):
        if self._gui is None:
            return
        if self._geom_menu is not None:
            self._geom_menu.clear()
        else:
            self._geom_menu = QtWidgets.QMenu("&Detector", self._gui)
            self._gui.menuBar().addMenu(self._geom_menu)

        geom_action_group = QtWidgets.QActionGroup(self._gui)

        '''
        # no geometry option, do we need this?
        none_action = QtWidgets.QAction('(None)', self._geom_menu)
        none_action.setCheckable(True)
        none_action.setActionGroup(geom_action_group)
        none_action.setChecked(True)
        none_action.triggered.connect(lambda checked, sender=none_action:\
                                 self._on_action_checked(checked, sender))
        self._geom_menu.addAction(none_action)

        # reset the actions map
        self._geom_action_map = {none_action: None}
        '''

        if self._available_geoms is None:
            return 

        first = True
        for geom in self._available_geoms:
            action = QtWidgets.QAction(geom.name(), self._geom_menu)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, sender=action:\
                                     self._on_action_checked(checked, sender))
            action.setActionGroup(geom_action_group)
            self._geom_menu.addAction(action)
            self._geom_action_map[action] = geom
            if first:
                action.setChecked(True)
                action.triggered.emit(True)
                first = False

    def _on_larsoft_services_loaded(self):
        """
        Creates list of geometries the user can select. Since LArSoftModule can
        only use either ICARUS or SBND services, these are mutually exclusive
        """
        if self._lsm.configured_for == 'sbnd':
            self._available_geoms = [
                geometry.sbnd(self._lsm.geom_service, self._lsm.det_prop_service,
                              self._lsm.det_clock_service, self._lsm.lar_prop_service)
            ]
        elif self._lsm.configured_for == 'icarus':
            self._available_geoms = [
                geometry.icarus(self._lsm.geom_service, self._lsm.det_prop_service,
                              self._lsm.det_clock_service, self._lsm.lar_prop_service)
            ]

        if self._time_range is not None:
            for geo in self._available_geoms:
                geo.override_time_range(self._time_range)

        for geo in self._available_geoms:
            geo.print_summary()

        # gui support is optional for this module. If not enabled, just
        # internally set the current geometry, and user will not be able to
        # change it at runtime.
        if self._gui is not None:
            self.populate_menu()
        else:
            self.current_geom = self._available_geoms[0]
            self.geometryChanged.emit()
            # also reload the event
            if self._gi is not None:
                self._gi.eventChanged.emit()

    def _on_action_checked(self, checked, sender):
        """ call back for QAction in geometry menu bar """
        geom = self._geom_action_map[sender]
        if geom != self.current_geom:
            self.current_geom = geom
            self.geometryChanged.emit()

        geom_name = 'None'
        if geom is not None:
            geom_name = geom.name()
        self._detector_label.setText(f'Detector: {geom_name}')
