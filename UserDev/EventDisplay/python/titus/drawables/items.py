"""
Graphics items for drawables in the wire views, whose hover area is a
fixed number of screen pixels at any zoom.
"""

from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

# hover distance from a line, in screen pixels
HOVER_PIXELS = 5


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
        self._points = points
        self._color = color if color is not None else (0, 0, 0)
        self._style = style
        self._shape_cache = (None, None)

        # Initialize a path:
        path = QtGui.QPainterPath()
        self.setPen(self._pen(2))

        # Fill the path:
        path.moveTo(points[0])
        for i in range(len(points)-1):
            path.lineTo(points[i+1])
        self.setPath(path)

    def _pen(self, width):
        return pg.mkPen(self._color, width=width, style=self._style)

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
        return super().boundingRect().united(self.shape().boundingRect())

    def hoverEnterEvent(self, e):
        QtWidgets.QGraphicsPathItem.hoverEnterEvent(self, e)
        self.setPen(self._pen(5))
        self.update()

    def hoverLeaveEvent(self, e):
        QtWidgets.QGraphicsPathItem.hoverLeaveEvent(self, e)
        self.setPen(self._pen(2))
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
