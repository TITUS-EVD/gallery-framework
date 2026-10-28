#!/usr/bin/env python3

"""
Provides the LArSoft services to the C++ drawers that still need them, loaded
the first time one asks (see Geometry's service_loader): startup only needs
the detector description files. Loading takes a while from cvmfs; the GUI
shows a wait cursor meanwhile. This module also provides a menu to select
the producer stages present in the loaded file, and maintains a dictionary of
LArSoft products present at each stage
"""

import time

from PyQt5 import QtWidgets, QtGui, QtCore

from .module import Module
from ..geometry import ServiceHandles


SUPPORTED_SERVICES = {
    'icarus': 'ICARUSservices',
    'sbnd': 'SBNDservices',
}


class LArSoftModule(Module):
    # emitted once the GUI is set up: the geometry can be created
    ready = QtCore.pyqtSignal()
    servicesLoaded = QtCore.pyqtSignal()
    stageChanged = QtCore.pyqtSignal()

    def __init__(self, service_name: str = '', time_range: int = None):
        super().__init__()
        service_name = service_name or 'sbnd'
        if service_name not in SUPPORTED_SERVICES:
            raise RuntimeError(f'{service_name} is not supported by LArSoft module')

        self._stage_label = QtWidgets.QLabel()
        self._stage_label.setText('Stage: All')
        self._stage_menu = None
        self._stage_action_map = {}
        self._current_stage = None

        # which services to load, and which detector description to use
        self.configured_for = service_name

        self._services_loaded = False

        self.time_range = time_range

        self.det_clock_service = None
        self.det_prop_service = None
        self.geom_service = None
        self.readout_service = None
        self.lar_prop_service = None
        self.auxDetGeometryCore = None

    @property
    def current_stage(self):
        return self._current_stage

    @property
    def services_loaded(self):
        return self._services_loaded

    def _initialize(self):
        self.populate_menu()
        self._gui.statusBar().insertPermanentWidget(0, self._stage_label)

        # once the event loop runs, as the other modules connect to the
        # geometry signals in their own _initialize
        QtCore.QTimer.singleShot(0, self._on_ready)

    def _on_ready(self):
        self.populate_menu()
        self.ready.emit()

    def populate_menu(self):
        if self._stage_menu is not None:
            self._stage_menu.clear()
        else:
            self._stage_menu = QtWidgets.QMenu("&Stage", self._gui)
            self._gui.menuBar().addMenu(self._stage_menu)

        stage_action_group = QtWidgets.QActionGroup(self._gui)
        available_stages = self._gi.get_stages()
        if available_stages is None:
            return

        for stage in available_stages:
            action = QtWidgets.QAction(stage, self._stage_menu)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, sender=action:\
                                     self._on_action_checked(checked, sender))
            action.setActionGroup(stage_action_group)
            self._stage_menu.addAction(action)
            self._stage_action_map[action] = stage
            if stage == 'all':
                self._current_stage = 'all'
                action.setChecked(True)
                action.triggered.emit(True)

    def _on_action_checked(self, checked, sender):
        """ call back for QAction in geometry menu bar """
        stage = self._stage_action_map[sender]
        if stage != self._current_stage:
            self._current_stage = stage
            self.stageChanged.emit()
        self._stage_label.setText(f'Stage: {stage}')

    def load_services(self):
        """
        Load the services (blocking) and return their handles. Used as the
        geometry's service_loader, so it runs the first time a drawer asks
        for a service.
        """
        if not self._services_loaded:
            self._show_loading(True)
            try:
                start = time.perf_counter()
                services = __import__(SUPPORTED_SERVICES[self.configured_for])
                self.det_clock_service = services.ServiceManager('DetectorClocks')
                self.det_prop_service = services.ServiceManager('DetectorProperties')
                self.geom_service = services.ServiceManager('Geometry')
                self.readout_service = services.ServiceManager('WireReadout')
                self.lar_prop_service = services.ServiceManager('LArProperties')
                self.auxDetGeometryCore = services.ServiceManager('AuxDetGeometry')
                print(f'LArSoft module loaded services in {time.perf_counter() - start:.1f} s')
            finally:
                self._show_loading(False)
            self._services_loaded = True
            self.servicesLoaded.emit()

        clocks = self.det_clock_service.DataForJob()
        return ServiceHandles(
            geometry_core=self.geom_service,
            detector_properties=self.det_prop_service.DataFor(clocks),
            wire_readout=self.readout_service,
            detector_clocks=clocks,
            lar_properties=self.lar_prop_service,
            aux_det_geometry=self.auxDetGeometryCore,
        )

    def _show_loading(self, loading):
        if self._gui is None:
            if loading:
                print('Loading LArSoft services...')
            return
        if loading:
            self._gui.statusBar().showMessage('Loading LArSoft services...')
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            QtWidgets.QApplication.processEvents()
        else:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._gui.statusBar().clearMessage()
