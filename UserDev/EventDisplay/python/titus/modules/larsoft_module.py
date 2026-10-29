#!/usr/bin/env python3

"""
The detector TITUS is configured for, and a menu to select the producer
stages present in the loaded file. TITUS no longer loads the LArSoft
services: geometry and timing come from the detector description files
(titus/detectors), and products are read by the titusio library.
"""

from PyQt5 import QtWidgets, QtCore

from .module import Module


SUPPORTED_DETECTORS = ('icarus', 'sbnd')


class LArSoftModule(Module):
    # emitted once the GUI is set up: the geometry can be created
    ready = QtCore.pyqtSignal()
    stageChanged = QtCore.pyqtSignal()

    def __init__(self, service_name: str = '', time_range: int = None):
        super().__init__()
        service_name = service_name or 'sbnd'
        if service_name not in SUPPORTED_DETECTORS:
            raise RuntimeError(f'{service_name} is not a supported detector ({", ".join(SUPPORTED_DETECTORS)})')

        self._stage_label = QtWidgets.QLabel()
        self._stage_label.setText('Stage: All')
        self._stage_menu = None
        self._stage_action_map = {}
        self._current_stage = None

        # which detector description to use
        self.configured_for = service_name

        self.time_range = time_range

    @property
    def current_stage(self):
        return self._current_stage

    def _initialize(self):
        self.populate_menu()
        self._gui.statusBar().insertPermanentWidget(0, self._stage_label)

        # once the event loop runs, as the other modules connect to the
        # geometry signals in their own _initialize
        QtCore.QTimer.singleShot(0, self._on_ready)

    def _on_ready(self):
        self.populate_menu()
        self.ready.emit()

    def on_file_change(self):
        # the stages of the new file
        self.populate_menu()

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
