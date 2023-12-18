#!/usr/bin/env python

""" Module base class for connecting GUI elements with gallery interface """

from PyQt5 import QtCore


class Module(QtCore.QObject):
    def __init__(self, gui=None, gallery_interface=None):
        super().__init__()
        self._gui = gui
        self._gi = gallery_interface
        self._parent = None
        self._central_widget = None
        self._dock_widgets = []
        self._active = True

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, val):
        self._parent = val

    def connect_gui(self, gui):
        """ Get references to GUI and its gallery interface """
        self._gui = gui
        self._gi = self._gui.gallery_interface

        self._gi.eventChanged.connect(self.on_event_change)
        self._gi.fileChanged.connect(self.on_file_change)

    def initialize(self):
        """
        Public wrapper for _initialize function. External code should call
        this method, and module sub-classes should override _initialize
        """
        if self._gui is None or self._gi is None:
            raise RuntimeError("Error: Module cannot be initialized until it is added to a GUI")
        
        self._initialize()
    
    def _initialize(self):
        pass

    # gallery interface event listeners
    def on_event_change(self):
        if self._active:
            self.update()

    def on_file_change(self):
        if self._active:
            self.update()

    # gui may call this function
    def update(self):
        pass

    def add_module(self, module):
        ''' allow sub-modules. Give a reference to this module & add to gui '''
        module.parent = self
        self._gui.add_module(module)

    """ Generic methods for showing/hiding all the GUI elements associated with this module """
    def activate(self):
        if self._central_widget is not None:
            if self._gui.centralWidget().indexOf(self._central_widget) == -1:
                self._gui.centralWidget().addWidget(self._central_widget)
            self._gui.centralWidget().setCurrentWidget(self._central_widget)
        for dw in self._dock_widgets:
            dw.show()

        if not self._active:
            self.update()
        self._active = True

    def deactivate(self):
        if self._central_widget is not None:
            self._gui.centralWidget().removeWidget(self._central_widget)
        for dw in self._dock_widgets:
            dw.hide()
        self._active = False

    def add_gallery_interface(self, gi):
        """ add gallery interface to this module without adding it to the GUI """
        if self._gi is not None:
            print("Warning: overwriting gallery interface on module")
        self._gi = gi
