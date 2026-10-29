"""
Graphics items for drawables in the wire views, whose hover area is a
fixed number of screen pixels at any zoom.
"""

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


class polyLine(QtWidgets.QGraphicsPathItem):
    '''
    A polyline (e.g. a projected track) that highlights on hover. The hover
    area is the line widened by HOVER_PIXELS on screen. Qt's default shape
    widens it in item coordinates (wires and ticks), so it grew with the
    zoom, and filled the region between a curved line and its chord.
    '''

    def __init__(self, points, color=None, style=QtCore.Qt.SolidLine, *args):
        super(polyLine, self).__init__()
        self.setAcceptHoverEvents(True)
        # the hover area: the line widened by HOVER_PIXELS (shape()), not the bounding box
        hover_on_shape(self)
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

    def hoverEnterEvent(self, e):
        QtWidgets.QGraphicsPathItem.hoverEnterEvent(self, e)
        self.setPen(self._pen(self._width + 3))
        self.update()

    def hoverLeaveEvent(self, e):
        QtWidgets.QGraphicsPathItem.hoverLeaveEvent(self, e)
        self.setPen(self._pen(self._width))
        self.update()


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
