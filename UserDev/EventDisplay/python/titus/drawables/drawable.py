"""
Drawable base class
Drawable objects receive a gallery interface and have analyze methods connected
to its signals. Drawables must also hold a reference to their parent module so
that they are only updated when their parent module is visible
"""
import sys

from PyQt5 import QtCore

from titus.display.projection import object_t0s
from titus.readers import ProductNotFound
from titus.readers.products import ASSOCIATIONS
from titus.readers.gallery import (association_candidates, file_products, read_event_product, string_vector,
                                   titusio)

_NULL_NAME = "null"

class Drawable:

    def __init__(self, gallery_interface, *args, **kwargs):
        self._gi = gallery_interface
        self._gi.eventChanged.connect(self.on_event_changed)
        self._gi.fileChanged.connect(self.on_file_changed)

        self._product_name = _NULL_NAME
        self._producer_name = _NULL_NAME
        self._process = None
        self._drawnObjects = []

        # some objects take a long time to draw
        self._cache = {}

        self.parent_module = None

    @property
    def product_name(self):
        return self._product_name

    @property
    def producer_name(self):
        return self._producer_name

    @producer_name.setter
    def producer_name(self, name):
        self._producer_name = name

    def analyze(self):
        if self._gi.event_handle() is None:
            return

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

    def init(self):
        self._process.initialize()

    '''
    # non-public wrappers
    # DEPRECATED since module is solely responsible for calling drawable.analyze
    # i.e., drawable should not update based on receive the signal from
    # gallery_interface but possibly useful for derived classes to do something
    # else on those signals
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
    '''

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


class ReaderDrawable(Drawable):
    """
    Drawable reading its products through the titusio library
    (titus.readers) instead of a C++ drawer with LArSoft services. Products
    of the current event are read in analyze() into self._data, one
    container per producer.
    """
    KIND = None  # titus.readers.KINDS name

    def __init__(self, gallery_interface, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._data = []
        # {(kind, tag): C++ type} of the current file, read when first needed
        self._file_products = None

    def init(self):
        pass

    def set_producer(self, producer):
        if producer is None:
            producer = _NULL_NAME
        if producer == self._producer_name:
            return
        self._producer_name = producer

    def producers(self):
        if self._producer_name == _NULL_NAME:
            return []
        if isinstance(self._producer_name, (list, tuple)):
            return list(self._producer_name)
        return [self._producer_name]

    def analyze(self):
        self.clearDrawnObjects()
        self._data = []
        ev = self._gi.event_handle()
        if ev is None:
            return
        for tag in self.producers():
            try:
                self._data.append(read_event_product(titusio(), ev, self.KIND, str(tag)))
            except ProductNotFound as e:
                print(f'Warning: {e}')
        self.on_data()

    def on_data(self):
        """ called after analyze() reads the products; derived classes may convert them """
        pass

    def on_file_changed(self):
        self._file_products = None

    def file_products(self):
        if self._file_products is None:
            ev = self._gi.event_handle()
            self._file_products = {} if ev is None else file_products(ev)
        return self._file_products

    def pfp_object_t0s(self, pfp_kind, tag, n_objects):
        '''
        T0 (ns, nan without) per object of the product tag through its
        PFParticle: pfp_kind is the PFParticle association kind (e.g.
        'pfp_tracks'), stored under the objects' tag
        '''
        pfp_objects = self.read_optional(pfp_kind, tag)
        pfp_t0s = t0s = None
        if pfp_objects is not None and len(pfp_objects):
            pfp_t0s = self.read_optional('pfp_t0s', pfp_objects.product_tags[pfp_objects.left_product[0]])
            if pfp_t0s is not None and len(pfp_t0s):
                t0s = self.read_optional('t0s', pfp_t0s.product_tags[pfp_t0s.right_product[0]])
        return object_t0s(n_objects, pfp_objects, pfp_t0s, t0s)

    def associated_hits(self, kind, tag):
        ''' (association to hits, the hits it points into) for the product tag, or (None, None) '''
        object_hits = self.read_optional(kind, tag)
        if object_hits is None or len(object_hits) == 0:
            return None, None
        hits = self.read_optional('hits', object_hits.product_tags[object_hits.right_product[0]])
        return (object_hits, hits) if hits is not None else (None, None)

    def read_optional(self, kind, tag):
        """
        Another product of the current event, e.g. an association: None if
        the file or event has no such product (full tag), or it can't be read
        """
        products = self.file_products()
        ev = self._gi.event_handle()
        if ev is None or not tag or (kind, tag) not in products:
            return None
        try:
            if kind in ASSOCIATIONS:
                return read_event_product(titusio(), ev, kind, tag, cpp_type=products[(kind, tag)],
                                          candidate_tags=string_vector(association_candidates(products)))
            return read_event_product(titusio(), ev, kind, tag)
        except Exception as e:
            print(f'Warning: could not read {kind} {tag}: {e}')
            return None


class LazyDrawable:
    '''
    Stands in for a drawable until a producer is selected, so that drawables
    building C++ drawers (and with them the LArSoft services) aren't created
    at startup. factory() creates the drawable.
    '''

    def __init__(self, factory, product_name):
        self._factory = factory
        self._drawable = None
        self.product_name = product_name
        self.parent_module = None

    @property
    def drawable(self):
        return self._drawable

    def set_producer(self, producer):
        if producer is None or producer == _NULL_NAME:
            if self._drawable is not None:
                self._drawable.set_producer(None)
            return
        if self._drawable is None:
            self._drawable = self._factory()
            self._drawable.parent_module = self.parent_module
        self._drawable.set_producer(producer)

    def analyze(self):
        if self._drawable is not None:
            self._drawable.analyze()

    def drawObjects(self):
        if self._drawable is not None:
            self._drawable.drawObjects()

    def clearDrawnObjects(self, obj_list=None):
        if self._drawable is not None:
            self._drawable.clearDrawnObjects()

    def __getattr__(self, name):
        # anything else is the drawable's, once it exists
        drawable = self.__dict__.get('_drawable')
        if drawable is None:
            raise AttributeError(f'{name!r}: drawable for {self.product_name} not created yet')
        return getattr(drawable, name)
