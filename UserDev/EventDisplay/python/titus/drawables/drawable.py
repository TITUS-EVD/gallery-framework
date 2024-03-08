"""
Drawable base class
Drawable objects receive a gallery interface and have analyze methods connected
to its signals. Drawables must also hold a reference to their parent module so
that they are only updated when their parent module is visible
"""
import sys

from PyQt5 import QtCore

_NULL_NAME = "null"

class Drawable:

    def __init__(self, gallery_interface, *args, **kwargs):
        self._gi = gallery_interface
        self._gi.eventChanged.connect(self._on_event_changed)
        self._gi.fileChanged.connect(self._on_file_changed)

        self._product_name = _NULL_NAME
        self._producer_name = _NULL_NAME
        self._process = None
        self._drawnObjects = []

        # some objects take a long time to draw
        self._cache = {}

        self.parent_module = None

    @property
    def product_name(self):
        return self._productName

    @property
    def producer_name(self):
        return self._producerName

    @producer_name.setter
    def producer_name(self, name):
        self._producer_name = name

    def analyze(self):
        self.clearDrawnObjects()

        if self._process is None:
            return

        if self._producer_name == _NULL_NAME:
            return
        
        self._process.analyze(self._gi.event_handle())

    def set_producer(self, producer):
        """
        Set the producer. This is a tag needed to get the art objects in the
        gallery classes. If there are objects drawn, they should be cleared
        """
        if producer == self._producer_name:
            return

        if producer is None:
            producer = _NULL_NAME

        try:
            # reco objects have setProducer method
            self._process.setProducer(str(producer))
        except AttributeError:
            # only reco classes have setProducer method. If this is a raw class,
            # allow set producer name to something else besides null name so that
            # it gets updated
            # raw objects have setInput method (maybe multiple producers)
            self._process.clearInput();
            if isinstance(producer, list):
                for p in producer:
                    self._process.addInput(p)
            else:
                self._process.setInput(str(producer))
        
        self._producer_name = producer
        self.analyze()

    def init(self):
        self._process.initialize()

    # non-public wrappers
    def _on_event_changed(self):
        if self.parent_module is None:
            return

        # can't think of a good reason to keep the previous event's objects
        # on a new event...
        self.clearDrawnObjects()

        if self._producer_name != _NULL_NAME and self.parent_module.is_active():
            self.drawObjects()

        self.on_event_changed()

    def _on_file_changed(self):
        self.on_file_changed()
        self._on_event_changed()

    def drawObjects(self):
        pass

    def clearDrawnObjects(self, obj_list=None):
        for obj in self._drawnObjects:
            obj.scene().removeItem(obj)
        self._drawnObjects = []

    # derived classes should override these
    def on_event_changed(self):
        pass

    def on_file_changed(self):
        pass

    def drawObjects(self):
        pass

    def enable(self):
        self._disabled = False

    def disable(self):
        self._disabled = True
