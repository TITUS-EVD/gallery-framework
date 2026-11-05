"""
Drawable base class
Drawable objects receive a gallery interface and have analyze methods connected
to its signals. Drawables must also hold a reference to their parent module so
that they are only updated when their parent module is visible
"""

import pyqtgraph as pg

from titus.display.layout import ViewLayout
from titus.display.projection import object_hit_tpcs, object_t0s, particle_tpcs
from titus.display.space_charge import SpaceChargeMap, displaced, in_true_space
from titus.readers import ProductNotFound
from titus.readers.products import ASSOCIATIONS
from titus.readers.gallery import (association_candidates, file_products, read_event_product, string_vector,
                                   titusio)

_NULL_NAME = "null"


def _view_box(item):
    """ the pyqtgraph ViewBox an item was added to (ViewBox.addItem), or None """
    parent = item.parentItem()
    while parent is not None:
        if isinstance(parent, pg.ViewBox):
            return parent
        parent = parent.parentItem()
    return None


def add_items(view, items):
    """
    Add graphics items to a pyqtgraph ViewBox, as ViewBox.addItem does, but
    update its auto range once instead of after every item: with auto range
    on, each update asks every item in the view for its bounds.
    """
    updating = view._updatingRange
    view._updatingRange = True
    try:
        for item in items:
            view.addItem(item)
    finally:
        view._updatingRange = updating
    view.updateAutoRange()


def remove_items(items):
    """
    Remove graphics items from their scene, and from the addedItems of the
    ViewBox they were added to: removing them from the scene only would
    leave them listed there, growing every event. One pass per ViewBox,
    since ViewBox.removeItem updates the auto range for every item.
    """
    by_view = {}
    for item in items:
        view = _view_box(item)
        if view is not None:
            by_view.setdefault(view, set()).add(id(item))
        if item.scene() is not None:
            item.scene().removeItem(item)
        if view is not None:
            item.setParentItem(None)
    for view, removed in by_view.items():
        view.addedItems[:] = [i for i in view.addedItems if id(i) not in removed]
        view.updateAutoRange()

class Drawable:

    def __init__(self, gallery_interface, *args, **kwargs):
        self._gi = gallery_interface
        self._gi.eventChanged.connect(self.on_event_changed)
        self._gi.fileChanged.connect(self.on_file_changed)

        self._product_name = _NULL_NAME
        self._producer_name = _NULL_NAME
        self._drawnObjects = []

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

    def add_to_view(self, view, items):
        """ draw items in a view (a TPC module wire view), and keep them to clear later """
        items = list(items)
        add_items(view._view, items)
        self._drawnObjects.extend(items)

    def disconnect_signals(self):
        ''' stop receiving the gallery interface's signals, once this drawable is removed '''
        for signal, slot in [(self._gi.eventChanged, self.on_event_changed),
                             (self._gi.fileChanged, self.on_file_changed)]:
            try:
                signal.disconnect(slot)
            except TypeError:
                pass

    def clearDrawnObjects(self, obj_list=None):
        remove_items(self._drawnObjects)
        self._drawnObjects = []

    # derived classes should override these
    def on_event_changed(self):
        pass

    def on_file_changed(self):
        pass

    def drawObjects(self):
        pass


class ReaderDrawable(Drawable):
    """
    Drawable reading its products through the titusio library
    (titus.readers) instead of a C++ drawer with LArSoft services. Products
    of the current event are read in analyze() into self._data, one
    container per producer.
    """
    KIND = None  # titus.readers.KINDS name
    # simulation products (e.g. MC truth): not expected in data, missing there without a warning
    SIMULATION = False

    def __init__(self, gallery_interface, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._data = []
        # {(kind, tag): C++ type} of the current file, read when first needed
        self._file_products = None

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
        # per producer: as read (positions shown in text), and whether drawn moved by space charge
        self._stored_data, self._space_charge_applied = [], []
        sce_map = self._sce_map = self.space_charge_map()
        for tag in self.producers():
            try:
                container = self._read(ev, self.KIND, str(tag))
            except ProductNotFound as e:
                if not (self.SIMULATION and self._gi.is_real_data()):
                    print(f'Warning: {e}')
                continue
            # objects where charge was deposited (MC truth, pandoraSCE): where the simulation moved it
            moved = sce_map is not None and in_true_space(self.KIND, tag)
            self._data.append(displaced(container, sce_map) if moved else container)
            self._stored_data.append(container)
            self._space_charge_applied.append(moved)
        self.on_data()

    def space_charge_map(self):
        '''
        The space charge map (titus.display.space_charge) the module draws
        objects in true space with, None to draw them as stored
        '''
        getter = getattr(getattr(self, '_module', None), 'space_charge_map', None)
        sce_map = getter() if callable(getter) else None
        return sce_map if isinstance(sce_map, SpaceChargeMap) else None

    def space_charge_applied(self, producer):
        ''' whether the objects of the producer (index in self._data) were moved by the space charge map '''
        applied = getattr(self, '_space_charge_applied', [])
        return producer < len(applied) and applied[producer]

    def _producer_of(self, container):
        return next((p for p, c in enumerate(self._data) if c is container), None)

    def stored(self, container):
        ''' a container of self._data as read, before space charge: the positions to show in text '''
        producer = self._producer_of(container)
        stored = getattr(self, '_stored_data', [])
        return stored[producer] if producer is not None and producer < len(stored) else container

    def space_charge_note(self, container):
        ''' a tooltip line for objects drawn moved by space charge (container: of self._data), else '' '''
        producer = self._producer_of(container)
        if producer is None or not self.space_charge_applied(producer):
            return ''
        return f'\ndrawn where space charge moved its charge ({self._sce_map.name})'

    def _product_cache(self):
        ''' the gallery interface's ProductCache, shared by all drawables (None without one) '''
        return getattr(self._gi, 'product_cache', None)

    def _read(self, ev, kind, tag, **kwargs):
        ''' read_event_product, through the shared cache: each product is read once per event '''
        cache = self._product_cache()
        read = lambda: read_event_product(titusio(), ev, kind, tag, **kwargs)  # noqa: E731
        return read() if cache is None else cache.get((kind, tag), read)

    def on_data(self):
        """ called after analyze() reads the products; derived classes may convert them """
        pass

    def on_file_changed(self):
        self._file_products = None

    def file_products(self):
        cache = self._product_cache()
        ev = self._gi.event_handle()
        if cache is not None:
            return cache.file_value('file_products', lambda: {} if ev is None else file_products(ev))
        if self._file_products is None:
            self._file_products = {} if ev is None else file_products(ev)
        return self._file_products

    def _association_candidates(self):
        ''' std::vector of the tags associations may point into, built once per file '''
        products = self.file_products()
        cache = self._product_cache()
        make = lambda: string_vector(association_candidates(products))  # noqa: E731
        return make() if cache is None else cache.file_value('association_candidates', make)

    def views(self):
        '''
        The TPC module's wire views to draw in: those shown by its view
        buttons (it redraws the products when it shows another)
        '''
        shown = getattr(self._module, '_view_shown', None)
        return [view for key, view in self._module._wire_views.items() if shown is None or shown(key)]

    def view_layout(self, view):
        ''' the layout of a wire view as currently shown, including ticks cut by "unite cathodes" '''
        removed = getattr(view, 'removed_ticks', None)
        return ViewLayout(self._geom, plane=view.plane(), cryostat=view.cryostat(),
                          removed_ticks=removed() if removed is not None else 0)

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

    def particle_tpcs(self, pfp_kind, tag, xyz):
        '''
        TPC per object (e.g. vertex, shower start at xyz) of the product tag,
        from its PFParticle's spacepoints and their hits (see
        titus.display.particle_tpcs); None without them
        '''
        pfp_objects = self.read_optional(pfp_kind, tag)
        if pfp_objects is None or len(pfp_objects) == 0:
            return None
        pfp_tag = pfp_objects.product_tags[pfp_objects.left_product[0]]
        pfp_spacepoints = self.read_optional('pfp_spacepoints', pfp_tag)
        if pfp_spacepoints is None or len(pfp_spacepoints) == 0:
            return None
        spacepoint_tag = pfp_spacepoints.product_tags[pfp_spacepoints.right_product[0]]
        spacepoints = self.read_optional('spacepoints', spacepoint_tag)
        spacepoint_hits, hits = self.associated_hits('spacepoint_hits', spacepoint_tag)
        if spacepoints is None or hits is None:
            return None
        return particle_tpcs(xyz, pfp_objects, pfp_spacepoints, spacepoints,
                             object_hit_tpcs(len(spacepoints), spacepoint_hits, hits))

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
                return self._read(ev, kind, tag, cpp_type=products[(kind, tag)],
                                  candidate_tags=self._association_candidates())
            return self._read(ev, kind, tag)
        except Exception as e:
            print(f'Warning: could not read {kind} {tag}: {e}')
            return None
