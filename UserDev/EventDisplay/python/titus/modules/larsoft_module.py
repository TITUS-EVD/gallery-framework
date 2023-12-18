#!/usr/bin/env python3

"""
Loads the LArSoft services. This can take a few seconds so we start it in a
separate thread and block the GUI until it's finished. This module also
provides a menu to select the producer stages present in the loaded file, and
maintains a dictionary of LArSoft products present at each stage
"""

from PyQt5 import QtWidgets, QtGui, QtCore

try:
    import SBNDservices as services
except ImportError:
    try:
        import ICARUSservices as services
    except ImportError:
        raise ImportError("LArSoft module could not import SBND or ICARUS services.")

from .module import Module


class LArSoftModule(Module):
    servicesLoaded = QtCore.pyqtSignal()
    stageChanged = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self._stage_label = QtWidgets.QLabel()
        self._stage_label.setText('Stage: All')
        self._stage_menu = None
        self._stage_action_map = {}
        self._current_stage = None

        # flag to indicate which services were loaded
        self.configured_for = 'sbnd'
        if 'icarus' in services.__name__.lower():
            self.configured_for = 'icarus'

        self._services_loaded = False

        self.det_clock_service = None
        self.det_prop_service = None
        self.geom_service = None
        self.lar_prop_service = None

        self._central_widget = QtWidgets.QWidget()
        waiting_layout = QtWidgets.QHBoxLayout()
        waiting_layout.addStretch(1)
        waiting_layout.addWidget(QtWidgets.QLabel('<h1>Starting LArSoft services...</h1>'))
        waiting_layout.addStretch(1)
        self._central_widget.setLayout(waiting_layout)

    @property
    def current_stage(self):
        return self._current_stage

    def _initialize(self):
        self.populate_menu()
        self._gui.statusBar().addPermanentWidget(self._stage_label)

        # start gui with this module's waiting widget
        self.activate()
        self._gui.setEnabled(False)

        # Start the thread after a small delay to give GUI chance to display
        QtCore.QTimer.singleShot(50, self._load_services)

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

    def _load_services(self):
        """ Services take a little while to load, so do it in a separate thread """
        if self._services_loaded:
            return

        self._thread = QtCore.QThread()
        self._worker = ServiceLoaderWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._on_service_worker_thread_finished)
        self._thread.start()

    def _on_service_worker_thread_finished(self):
        self._services_loaded = True

        self.det_clock_service = self._worker.det_clock_service
        self.det_prop_service = self._worker.det_prop_service
        self.geom_service = self._worker.geom_service
        self.lar_prop_service = self._worker.lar_prop_service

        self._gui.setEnabled(True)
        self._worker.deleteLater()
        self._thread.deleteLater()
        
        self.deactivate()
        self._central_widget.deleteLater()
        self.populate_menu()

        self.servicesLoaded.emit()


class ServiceLoaderWorker(QtCore.QObject):
    """ Load LArSoft services in this thread. Main thread will get them once loaded """
    finished = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()

    def run(self):
        self.det_clock_service = services.ServiceManager('DetectorClocks')
        self.det_prop_service = services.ServiceManager('DetectorProperties')
        self.geom_service = services.ServiceManager('Geometry')
        self.lar_prop_service = services.ServiceManager('LArProperties')

        # TODO is this necessary?
        self.det_prop_service.DataFor(self.det_clock_service.DataForJob())

        print('LArSoft module loaded services')

        self.finished.emit()
