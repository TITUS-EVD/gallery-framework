#!/usr/bin/env python3

"""
This module adds UI elements for selecting different geometries. Geometries
come from the detector description files; the LArSoft module only provides
the services, loaded when a drawer first needs them.

Each file is drawn with the description of the GDML it was made with (read
from its configuration, GalleryInterface.file_gdml), e.g. SBND sbnd_v02_04
or sbnd_v02_06, unless one is given on the command line (--gdml).
"""

import os

from PyQt5 import QtWidgets, QtGui, QtCore

from .module import Module
from ..geometry import Geometry, find_gdml, load_detector, load_display_settings

_WARNING = "\033[91m WARNING\033[0m"


class GeometryModule(Module):
    # a new Geometry: views are built for it
    geometryChanged = QtCore.pyqtSignal()
    # the current Geometry drawn with another description of the same layout
    # (e.g. the GDML of the file opened): positions changed, the views stay
    detectorChanged = QtCore.pyqtSignal()

    def __init__(self, larsoft_module, gdml=None):
        """
        gdml: the GDML name (e.g. 'sbnd_v02_06') or detector description
        file to draw every file with; None: each file's own
        """
        super().__init__()

        self._lsm = larsoft_module
        self._lsm.ready.connect(self._create_geometries)
        self._pinned_gdml = gdml
        self._following_files = False

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

    def add_gallery_interface(self, gi):
        super().add_gallery_interface(gi)
        self._follow_files()

    def connect_gui(self, gui):
        super().connect_gui(gui)
        self._follow_files()

    def _follow_files(self):
        '''
        switch geometry when a file is opened: connected before the view
        modules, which then read the new file with it
        '''
        if not self._following_files and self._gi is not None:
            self._gi.fileChanged.connect(self._on_file_opened)
            self._following_files = True

    def _load_geometry(self, detector_name):
        ''' the geometry drawn first: --gdml, else the detector's default description '''
        gdml = self._pinned_gdml
        if gdml is not None and gdml.endswith(('.json', '.json.gz')) and os.path.exists(gdml):
            detector = load_detector(gdml)
        else:
            detector = load_detector(detector_name, gdml)
        return Geometry(detector, load_display_settings(detector_name))

    def _on_file_opened(self):
        self.follow_gdml(self._gi.file_gdml, self._gi.current_file)

    def follow_gdml(self, gdml, file=''):
        '''
        Draw with the description of the GDML a file was made with: the
        current geometry switches to it in place (Geometry.set_detector) if
        TITUS has one with the same layout. Returns the GDML drawn.
        '''
        geom = self.current_geom
        if geom is None:
            return None
        name = os.path.basename(file) or 'the file'
        if gdml is None:
            print(f'Geometry: {name} does not say which GDML it was made with, drawn with {geom.gdml()}')
            return geom.gdml()
        if self._pinned_gdml is not None:
            if find_gdml(geom.name(), gdml) != geom.gdml():
                print(f'{_WARNING} {name} was made with {gdml}, drawn with {geom.gdml()} (--gdml)')
            return geom.gdml()
        shipped = find_gdml(geom.name(), gdml)
        if shipped is None:
            print(f'{_WARNING} {name} was made with {gdml}, which TITUS has no description of: drawn with '
                  f'{geom.gdml()} (tools/dump_detector.py in its release makes one)')
            return geom.gdml()
        if shipped == geom.gdml():
            return shipped
        detector = load_detector(geom.name(), shipped)
        if not geom.can_switch_to(detector):
            print(f'{_WARNING} {name} was made with {shipped}, whose layout differs from {geom.gdml()}: '
                  f'restart TITUS with --gdml {shipped} to draw it')
            return geom.gdml()
        previous = geom.gdml()
        geom.set_detector(detector)
        print(f'Geometry: {name} was made with {gdml}, drawn with {shipped} (was {previous})')
        self._show_detector(geom)
        self.detectorChanged.emit()
        return shipped

    def _show_detector(self, geom):
        self._detector_label.setText(f'Detector: {geom.name()} ({geom.gdml()})' if geom is not None
                                     else 'Detector: None')

    def _create_geometries(self):
        """
        Creates the list of geometries the user can select: the detector
        chosen on the command line (LArSoftModule.configured_for)
        """
        # geometry and timing come from the detector description files
        self._available_geoms = [self._load_geometry(self._lsm.configured_for)]

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

        self._show_detector(geom)
