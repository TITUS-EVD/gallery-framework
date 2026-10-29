import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from titus.display import hit_positions
from titus.drawables.drawable import ReaderDrawable


class HitCollection(QtWidgets.QGraphicsItem):
    '''
    All the hits of a view in one item: boxes (x, y, width, height in view
    coordinates) filled black with an opacity each (alpha), or with a color
    each (colors, (n, 4) RGBA). One item instead of one per hit keeps drawing
    and clearing an event fast. The box under the mouse is outlined, with
    the other boxes of its group if given (e.g. a cluster's hits), and its
    tooltip comes from tooltip(index).
    '''

    def __init__(self, x, y, width, height, alpha=None, tooltip=None, colors=None, group=None):
        super().__init__()
        self.x0, self.y0 = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.x1, self.y1 = self.x0 + width, self.y0 + height
        if colors is None:
            colors = np.zeros((len(self.x0), 4), dtype=int)
            colors[:, 3] = 255 if alpha is None else alpha
        self.colors = np.clip(np.asarray(colors, dtype=int).reshape(-1, 4), 0, 255)
        self.group = None if group is None else np.asarray(group)
        self._tooltip = tooltip if tooltip is not None else (lambda index: '')
        self._hovered = None
        self.setAcceptHoverEvents(True)
        # rects grouped by color, so that painting is one drawRects per color
        unique, inverse = np.unique(self.colors, axis=0, return_inverse=True)
        self._groups = [(QtGui.QBrush(pg.mkColor(*(int(c) for c in color))),
                         [self.rect(i) for i in np.flatnonzero(inverse.ravel() == k)])
                        for k, color in enumerate(unique)]
        if len(self.x0):
            self._bounds = QtCore.QRectF(self.x0.min(), self.y0.min(),
                                         self.x1.max() - self.x0.min(), self.y1.max() - self.y0.min())
        else:
            self._bounds = QtCore.QRectF()
        self._hover_pen = pg.mkPen('r', width=2)
        self._hover_pen.setCosmetic(True)

    def __len__(self):
        return len(self.x0)

    def rect(self, index):
        return QtCore.QRectF(self.x0[index], self.y0[index], self.x1[index] - self.x0[index],
                             self.y1[index] - self.y0[index])

    def boundingRect(self):
        return self._bounds

    def paint(self, painter, option, widget):
        painter.setPen(QtCore.Qt.NoPen)
        for brush, rects in self._groups:
            painter.setBrush(brush)
            painter.drawRects(rects)
        if self._hovered is not None:
            painter.setPen(self._hover_pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRects([self.rect(i) for i in self.highlighted()])

    def highlighted(self):
        ''' rows outlined: the hovered box, and the rest of its group '''
        if self._hovered is None:
            return []
        if self.group is None:
            return [self._hovered]
        return list(np.flatnonzero(self.group == self.group[self._hovered]))

    def index_at(self, x, y):
        ''' the hit whose box contains the view point (the last drawn on top), or None '''
        inside = np.flatnonzero((self.x0 <= x) & (x < self.x1) & (self.y0 <= y) & (y < self.y1))
        return int(inside[-1]) if len(inside) else None

    def _hover(self, index):
        if index == self._hovered or (index is not None and self._hovered is not None and self.group is not None
                                      and self.group[index] == self.group[self._hovered]):
            return
        self._hovered = index
        self.setToolTip(self._tooltip(index) if index is not None else '')
        self.update()

    def hoverMoveEvent(self, e):
        self._hover(self.index_at(e.pos().x(), e.pos().y()))

    def hoverLeaveEvent(self, e):
        self._hover(None)


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
            for view in self.views():
                self.drawHitList(view, hits)

    def drawHitList(self, view, hits):
        if len(hits) == 0:
            return

        geom = self._geom
        layout = self.view_layout(view)
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
                y_start[rows] = layout.tick_to_view(block.plane_index, hits.start_tick[index], hide_removed=False)
                y_end[rows] = layout.tick_to_view(block.plane_index, hits.end_tick[index] + 1, hide_removed=False)
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

        rows = positions.index
        collection = HitCollection(positions.x - 0.5, bottom, np.ones(len(positions)), height, opacity,
                                   tooltip=lambda i: self.genToolTip(hits, rows[i]))
        self._drawnObjects.append(collection)
        view._view.addItem(collection)

    def genToolTip(self, hits, row):
        return 'Time: {time:0.1f}\nRMS: {rms:0.1f}\nIntegral: {integral:0.1f}'.format(
            time=hits.peak_time[row],
            rms=hits.rms[row],
            integral=hits.integral[row])
