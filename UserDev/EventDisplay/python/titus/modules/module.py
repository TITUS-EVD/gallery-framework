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
        self._dock_widgets = set()
        self._active = True

        self._drawables = set()
        self._thread_pool = QtCore.QThreadPool()

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, val):
        self._parent = val

    def is_active(self):
        return self._active

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
            self._update()

    def on_file_change(self):
        if self._active:
            self._update()

    def _update(self):
        for d in self._drawables:
            # worker = DrawableWorker(d)
            # worker.signals.finished.connect(d.drawObjects)
            # self._thread_pool.start(worker)
            d.analyze()
            d.drawObjects()
        self.update()

    # gui may call this function
    def update(self):
        pass

    def add_module(self, module):
        ''' allow sub-modules. Give a reference to this module & add to gui '''
        module.parent = self
        self._gui.add_module(module)

    def activate(self):
        """ Generic methods for showing/hiding all the GUI elements associated with this module """
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
        self._active = False
        if self._central_widget is not None:
            self._gui.centralWidget().removeWidget(self._central_widget)
        for dw in self._dock_widgets:
            dw.hide()

    def add_gallery_interface(self, gi):
        """ add gallery interface to this module without adding it to the GUI """
        if self._gi is not None:
            print("Warning: overwriting gallery interface on module")
        self._gi = gi

    def register_drawable(self, drawable):
        drawable.parent_module = self
        self._drawables.add(drawable)
        return drawable

    def remove_drawable(self, drawable):
        if drawable is None:
            return
        drawable.clearDrawnObjects() 
        self._drawables.remove(drawable)

    # TODO need to rework widgets so that this kind of event forwarding to 
    # gui isn't necessary
    def keyPressEvent(self, e):
        return self._gui.keyPressEvent(e)


class DrawableWorker(QtCore.QRunnable):
    def __init__(self, drawable):
        super().__init__()
        self._drawable = drawable
        self.signals = WorkerSignals()

    def run(self):
        self._drawable.analyze()


class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
