"""
Graphics items for drawables in the wire views, whose hover area is a
fixed number of screen pixels at any zoom.
"""

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

# hover distance from a line, in screen pixels
HOVER_PIXELS = 5


def hover_on_shape(item):
    '''
    Make hover and tooltips follow the item's shape: items in a pyqtgraph
    ViewBox are clipped to it, and Qt then tests a clipped item's bounding
    rectangle (its clip path) instead of its shape, so the hover area of a
    line at an angle filled its whole bounding box. With ItemClipsToShape
    the clip path is the shape. The shape must contain what is painted.
    '''
    item.setFlag(QtWidgets.QGraphicsItem.ItemClipsToShape)
    return item

# the color of selected objects (e.g. the MC truth dock's interaction and particles)
SELECTED_COLOR = (255, 0, 255)


def _device_transform(item):
    ''' item coordinates -> screen pixels, or None outside a view '''
    scene = item.scene()
    if scene is None or not scene.views():
        return None
    return item.deviceTransform(scene.views()[0].viewportTransform())


class _HoverArea(QtWidgets.QGraphicsItem):
    '''
    The hover area and tooltip of a polyLine, as an invisible child item: it
    is clipped to the line's shape (hover_on_shape), so hover and tooltips
    follow the line, and having no contents it costs nothing to paint. The
    line itself paints unclipped: clipping each line to its shape made
    repaints several times slower.
    '''

    def __init__(self, line):
        super().__init__(line)
        self._line = line
        self.setFlag(QtWidgets.QGraphicsItem.ItemHasNoContents)
        hover_on_shape(self)
        self.setAcceptHoverEvents(True)

    def boundingRect(self):
        return self._line.boundingRect()

    def shape(self):
        return self._line.shape()

    def paint(self, painter, option, widget=None):
        pass

    def hoverEnterEvent(self, e):
        super().hoverEnterEvent(e)
        self._line.set_hovered(True)

    def hoverLeaveEvent(self, e):
        super().hoverLeaveEvent(e)
        self._line.set_hovered(False)


class polyLine(QtWidgets.QGraphicsPathItem):
    '''
    A polyline (e.g. a projected track) that highlights on hover. The hover
    area is the line widened by HOVER_PIXELS on screen (shape(); Qt's
    default shape widens it in item coordinates, so it grew with the zoom,
    and filled the region between a curved line and its chord), handled by
    a child _HoverArea, which also holds the tooltip.
    '''

    def __init__(self, points, color=None, style=QtCore.Qt.SolidLine, *args):
        super(polyLine, self).__init__()
        self._hover = _HoverArea(self)
        self._points = points
        self._color = color if color is not None else (0, 0, 0)
        self._base_color = self._color
        self._width = 2
        self._style = style
        self._shape_cache = (None, None)

        # Initialize a path:
        path = QtGui.QPainterPath()
        self.setPen(self._pen(self._width))

        # Fill the path:
        path.moveTo(points[0])
        for i in range(len(points)-1):
            path.lineTo(points[i+1])
        self.setPath(path)

    def _pen(self, width):
        return pg.mkPen(self._color, width=width, style=self._style)

    def set_selected(self, selected):
        ''' drawn thicker in SELECTED_COLOR, above other lines '''
        self._color = SELECTED_COLOR if selected else self._base_color
        self._width = 4 if selected else 2
        self.setZValue(1 if selected else 0)
        self.setPen(self._pen(self._width))

    def is_selected(self):
        return self._color == SELECTED_COLOR

    def shape(self):
        transform = _device_transform(self)
        if transform is None:
            return super().shape()
        # the hover area only depends on the scale and rotation, not on panning
        key = (transform.m11(), transform.m12(), transform.m21(), transform.m22())
        if self._shape_cache[0] != key:
            inverse, invertible = transform.inverted()
            if not invertible:
                return super().shape()
            stroker = QtGui.QPainterPathStroker()
            stroker.setWidth(2 * HOVER_PIXELS)
            stroker.setCapStyle(QtCore.Qt.RoundCap)
            stroker.setJoinStyle(QtCore.Qt.RoundJoin)
            # widen on screen, then back to item coordinates
            on_screen = stroker.createStroke(transform.map(self.path()))
            self._shape_cache = (key, inverse.map(on_screen))
        return self._shape_cache[1]

    def boundingRect(self):
        # the path's bounds widened by the hover distance, without stroking
        # the path: views ask every item for its bounds whenever one is added
        rect = super().boundingRect()
        transform = _device_transform(self)
        if transform is None:
            return rect
        inverse, invertible = transform.inverted()
        if not invertible:
            return rect
        pad_x = HOVER_PIXELS * (inverse.m11() ** 2 + inverse.m21() ** 2) ** 0.5
        pad_y = HOVER_PIXELS * (inverse.m12() ** 2 + inverse.m22() ** 2) ** 0.5
        return rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)

    def set_hovered(self, hovered):
        self.setPen(self._pen(self._width + 3 if hovered else self._width))

    def setToolTip(self, text):
        # on the hover area: the line's own bounding box would show it far from the line
        self._hover.setToolTip(text)

    def toolTip(self):
        return self._hover.toolTip()

    def hover_area(self):
        return self._hover


class Marker(QtWidgets.QGraphicsEllipseItem):
    '''
    A circle of fixed size on screen (radius in pixels) at a point, e.g. a
    vertex. It ignores the view's zoom, so its hover area does too.
    '''

    def __init__(self, x, y, radius=5, color=(0, 255, 255), filled=True):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        self._color = color
        self._filled = filled
        self._style(1)

    def _style(self, width):
        self.setPen(pg.mkPen(self._color if not self._filled or width > 1 else (0, 0, 0), width=width))
        self.setBrush(pg.mkBrush(self._color) if self._filled else QtGui.QBrush(QtCore.Qt.NoBrush))

    def hoverEnterEvent(self, e):
        super().hoverEnterEvent(e)
        self._style(3)

    def hoverLeaveEvent(self, e):
        super().hoverLeaveEvent(e)
        self._style(1)


class CrossMarker(QtWidgets.QGraphicsPathItem):
    '''
    An X of fixed size on screen (half-size in pixels) at a point, e.g. the
    neutrino interaction vertex. Like Marker, it ignores the view's zoom.
    '''

    COLOR = (250, 128, 114)
    SIZE, SELECTED_SIZE = 8, 11

    def __init__(self, x, y, size=SIZE, color=COLOR, width=3):
        super().__init__(self._cross(size))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        self._color, self._width = color, width
        self.setPen(pg.mkPen(color, width=width))

    @staticmethod
    def _cross(size):
        path = QtGui.QPainterPath()
        path.moveTo(-size, -size)
        path.lineTo(size, size)
        path.moveTo(-size, size)
        path.lineTo(size, -size)
        return path

    def set_selected(self, selected):
        ''' larger, in SELECTED_COLOR, above the other crosses '''
        self.prepareGeometryChange()
        self.setPath(self._cross(self.SELECTED_SIZE if selected else self.SIZE))
        self._color = SELECTED_COLOR if selected else self.COLOR
        self.setZValue(2 if selected else 1)
        self.setPen(pg.mkPen(self._color, width=self._width))

    def is_selected(self):
        return self._color == SELECTED_COLOR

    def shape(self):
        # hover anywhere in the X's square, not only on its strokes
        path = QtGui.QPainterPath()
        path.addRect(self.path().boundingRect())
        return path

    def hoverEnterEvent(self, e):
        super().hoverEnterEvent(e)
        self.setPen(pg.mkPen(self._color, width=self._width + 2))

    def hoverLeaveEvent(self, e):
        super().hoverLeaveEvent(e)
        self.setPen(pg.mkPen(self._color, width=self._width))


class TrackLine:
    '''
    One polyline of a TrackCollection (e.g. a projected track segment):
    points (n, 2) in view coordinates, color, pen style, tooltip, and key
    (e.g. (producer, row)). Selected lines are drawn thicker in
    SELECTED_COLOR, above the others.
    '''

    def __init__(self, points, color=(0, 0, 0), style=QtCore.Qt.SolidLine, tooltip='', key=None):
        self.points = np.asarray(points, dtype=float).reshape(-1, 2)
        self.color = color
        self.style = style
        self.track_key = key
        self._tooltip = tooltip
        self._selected = False
        self._pens = {}
        self.collection = None  # set when added to one
        self.polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in self.points])

    def toolTip(self):
        return self._tooltip

    def setToolTip(self, text):
        self._tooltip = text

    def width(self):
        return 4 if self._selected else 2

    def pen(self, hovered=False):
        key = (self._selected, hovered)
        if key not in self._pens:
            color = SELECTED_COLOR if self._selected else self.color
            self._pens[key] = pg.mkPen(color, width=self.width() + (3 if hovered else 0), style=self.style)
        return self._pens[key]

    def set_selected(self, selected):
        if selected != self._selected:
            self._selected = selected
            if self.collection is not None:
                self.collection.update()

    def is_selected(self):
        return self._selected

    def path(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.polygon)
        return path


class TrackCollection(QtWidgets.QGraphicsItem):
    '''
    Many TrackLines in one graphics item: each line drawn with its own pen,
    and hover handled here, by the distance from the mouse to every segment
    in screen pixels (the line within HOVER_PIXELS, highlighted, with its
    tooltip). With one item per line, every mouse move asked each of them
    for its bounding box, in Python: one item keeps that cost independent
    of the number of tracks. Lines outside the repainted area are skipped.
    '''

    def __init__(self, lines):
        super().__init__()
        self.lines = list(lines)
        for line in self.lines:
            line.collection = self
        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemUsesExtendedStyleOption)
        self._hovered = None
        # segments (x0, y0, x1, y1) and the line each belongs to
        segments = [np.hstack([line.points[:-1], line.points[1:]]) for line in self.lines if len(line.points) > 1]
        owners = [np.full(len(line.points) - 1, i) for i, line in enumerate(self.lines) if len(line.points) > 1]
        self._segments = np.vstack(segments) if segments else np.zeros((0, 4))
        self._owners = np.concatenate(owners) if owners else np.zeros(0, dtype=int)
        bounds = [(line.points.min(axis=0), line.points.max(axis=0)) for line in self.lines if len(line.points)]
        self._line_min = np.array([b[0] for b in bounds]).reshape(-1, 2)
        self._line_max = np.array([b[1] for b in bounds]).reshape(-1, 2)
        self._rect = (QtCore.QRectF(QtCore.QPointF(*self._line_min.min(axis=0)), QtCore.QPointF(*self._line_max.max(axis=0)))
                      if len(bounds) else QtCore.QRectF())

    def __len__(self):
        return len(self.lines)

    def _padding(self):
        ''' HOVER_PIXELS (and the widest pen) in item units, along x and y '''
        transform = _device_transform(self)
        if transform is None:
            return 0., 0.
        inverse, invertible = transform.inverted()
        if not invertible:
            return 0., 0.
        pixels = HOVER_PIXELS + 4
        return (pixels * (inverse.m11() ** 2 + inverse.m21() ** 2) ** 0.5,
                pixels * (inverse.m12() ** 2 + inverse.m22() ** 2) ** 0.5)

    def boundingRect(self):
        pad_x, pad_y = self._padding()
        return self._rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)

    def paint(self, painter, option, widget=None):
        exposed = option.exposedRect
        pad_x, pad_y = self._padding()
        visible = ((self._line_max[:, 0] >= exposed.left() - pad_x) & (self._line_min[:, 0] <= exposed.right() + pad_x)
                   & (self._line_max[:, 1] >= exposed.top() - pad_y) & (self._line_min[:, 1] <= exposed.bottom() + pad_y))
        drawable = [line for line in self.lines if len(line.points)]
        order = sorted((i for i in np.flatnonzero(visible)), key=lambda i: drawable[i].is_selected())
        for i in order:
            line = drawable[i]
            painter.setPen(line.pen(hovered=line is self._hovered_line()))
            painter.drawPolyline(line.polygon)

    def _hovered_line(self):
        return None if self._hovered is None else self.lines[self._hovered]

    def line_at(self, pos):
        ''' the line within HOVER_PIXELS of an item point (on screen), or None '''
        transform = _device_transform(self)
        if transform is None or len(self._segments) == 0:
            return None
        m = np.array([[transform.m11(), transform.m21()], [transform.m12(), transform.m22()]])
        offset = np.array([transform.dx(), transform.dy()])
        a = self._segments[:, :2] @ m.T + offset
        b = self._segments[:, 2:] @ m.T + offset
        p = np.array([pos.x(), pos.y()]) @ m.T + offset
        ab = b - a
        length2 = np.einsum('ij,ij->i', ab, ab)
        t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.where(length2 > 0, length2, 1), 0, 1)
        distance = np.linalg.norm(a + t[:, None] * ab - p, axis=1)
        nearest = int(np.argmin(distance))
        if distance[nearest] > HOVER_PIXELS:
            return None
        # of several lines that close, the selected one (drawn on top)
        close = self._owners[distance <= HOVER_PIXELS]
        selected = [i for i in close if self.lines[i].is_selected()]
        return int(selected[0]) if selected else int(self._owners[nearest])

    def _hover(self, index):
        if index == self._hovered:
            return
        self._hovered = index
        self.setToolTip(self.lines[index].toolTip() if index is not None else '')
        self.update()

    def hoverMoveEvent(self, e):
        self._hover(self.line_at(e.pos()))

    def hoverLeaveEvent(self, e):
        self._hover(None)
