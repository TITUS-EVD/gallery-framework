import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from titus.display import ViewLayout, hit_positions
from titus.drawables.drawable import ReaderDrawable


class HitBoxCollection(QtWidgets.QGraphicsItem):
    '''
    A collection of HitBox classes to be drawn at the same time.
    '''

    def __init__(self):
        super().__init__()
        self._rects = []

    def add_hitbox(self, hitbox):
        self._rects.append(hitbox)

    def add_hitboxes(self, rects):
        self._rects = rects
    
    def boundingRect(self):
        # required by QGraphicsItem; the boxes are child items
        return self.childrenBoundingRect() if self._rects else QtCore.QRectF()

    def paint(self, painter, option, widget):
        # This method is intentionally left empty because QGraphicsRectItem
        # will handle their own painting.
        pass


class HitBox(QtWidgets.QGraphicsRectItem):
    '''
    Represents a single Hit
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.setAcceptHoverEvents(True)
        self._isHighlighted = False

    def hoverEnterEvent(self, e):
        self.setPen(pg.mkPen('r', width=2))

    def hoverLeaveEvent(self, e):
        self.setPen(pg.mkPen(None))


class Hit(ReaderDrawable):
    '''
    Hits as boxes on the wire views, one wire wide and centered on the peak
    time, +- RMS tall (start to end tick for fasthit). Positions come from
    titus.display, the same layout as the wire images.
    '''
    KIND = 'hits'

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'hit'
        self._geom = geom

        # this product is only drawn when TPC module is active, and it needs to
        # access the GUI elements in the TPC module class.
        self._module = tpc_module

    def drawObjects(self):
        for hits in self._data:
            for view in self._module._wire_views.values():
                self.drawHitList(view, hits)

    def drawHitList(self, view, hits):
        if len(hits) == 0:
            return

        geom = self._geom
        layout = ViewLayout(geom, plane=view.plane(), cryostat=view.cryostat())
        positions = hit_positions(layout, hits, time_offset_ticks=geom.timeOffsetTicks(view.plane()))
        if len(positions) == 0:
            return

        if 'fasthit' in str(self._producer_name):
            # SBND commissioning fasthit finder: box from start to end tick
            y_start = np.empty(len(positions))
            y_end = np.empty(len(positions))
            for block in layout.blocks:
                rows = hits.on_plane(*geom.detector.planes[block.plane_index].id)[positions.index]
                index = positions.index[rows]
                y_start[rows] = layout.tick_to_view(block.plane_index, hits.start_tick[index])
                y_end[rows] = layout.tick_to_view(block.plane_index, hits.end_tick[index] + 1)
            bottom, height = np.minimum(y_start, y_end), np.abs(y_end - y_start)
        else:
            bottom, height = positions.y - positions.half_height, 2 * positions.half_height

        # opacity by charge, relative to the largest on the same plane
        integral = hits.integral[positions.index]
        max_charge = np.ones(len(positions))
        planes = np.stack([hits.cryostat, hits.tpc, hits.plane], axis=1)[positions.index]
        for plane in np.unique(planes, axis=0):
            same = np.all(planes == plane, axis=1)
            max_charge[same] = max(integral[same].max(), 1e-12)
        opacity = (128 * integral / max_charge).astype(int) + 127

        hit_collection = HitBoxCollection()
        rects = []
        for i, row in enumerate(positions.index):
            r = HitBox(positions.x[i] - 0.5, bottom[i], 1, height[i], hit_collection)
            r.setPen(pg.mkPen(None))
            r.setBrush(pg.mkColor(0, 0, 0, int(np.clip(opacity[i], 0, 255))))
            r.setToolTip(self.genToolTip(hits, row))
            rects.append(r)

        hit_collection.add_hitboxes(rects)
        self._drawnObjects.append(hit_collection)
        view._view.addItem(hit_collection)

    def genToolTip(self, hits, row):
        return 'Time: {time:0.1f}\nRMS: {rms:0.1f}\nIntegral: {integral:0.1f}'.format(
            time=hits.peak_time[row],
            rms=hits.rms[row],
            integral=hits.integral[row])
