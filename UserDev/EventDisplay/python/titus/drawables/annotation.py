"""
Annotations of MC truth particles in the wire views (the MC truth dock's
Annotate boxes): the particle's projected path in solid white, over the
tracks (unless only the names are shown), and its name above the line,
along it. A straight line standing in for a trajectory not known (e.g. a
neutral particle's) is dotted; the incoming neutrino's (the dock's Neutrino
path) is dashed, with an arrowhead at the vertex.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from titus.drawables.items import render_pen
from titus.gui.widgets import MovableLabel

ANNOTATION_COLOR = (255, 255, 255)
LINE_WIDTH = 2  # pixels
# dash and gap lengths, in line widths (Qt's DashLine is 4, 2): the neutrino's dashed line,
# and the dotted straight lines of the particles without a trajectory (e.g. neutral)
DASH_PATTERN = (4, 4)
DOT_PATTERN = (1, 2)
LINE_PATTERNS = {'solid': None, 'dashed': DASH_PATTERN, 'dotted': DOT_PATTERN}
# arrowheads: length and half width, in line widths
ARROW_LENGTH = 6
ARROW_HALF_WIDTH = 2.5
FONT_POINT_SIZE = 12
# above the tracks (TrackCollection) and their highlight
Z_VALUE = 10


def label_place(points):
    '''
    (point, direction) of a path's label, in view coordinates: half way along
    the path, along its direction there (over the middle fifth, steadier than
    one step), pointing to +x so the text reads left to right
    '''
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.], np.cumsum(lengths)])
    total = along[-1]

    def at(fraction):
        return np.array([np.interp(fraction * total, along, points[:, a]) for a in (0, 1)])

    middle = at(0.5)
    direction = at(0.6) - at(0.4)
    if np.allclose(direction, 0):
        direction = points[-1] - points[0]
    if np.allclose(direction, 0):
        direction = np.array([1., 0.])
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction
    return middle, direction


class AnnotationPath(QtWidgets.QGraphicsPathItem):
    '''
    a particle's path, its pixel width and dashes widened in images rendered
    larger (titus.drawables.items). arrow: (tip, from) view points, an
    arrowhead at tip pointing away from from, its size in pixels whatever
    the zoom
    '''

    def __init__(self, path, arrow=None):
        super().__init__(path)
        self.arrow = arrow

    def paint(self, painter, option, widget=None):
        pen = render_pen(self.pen())
        painter.setPen(pen)
        painter.setBrush(self.brush())
        painter.drawPath(self.path())
        if self.arrow is not None:
            head = arrowhead(painter.transform(), *self.arrow, max(pen.widthF(), 1.))
            if head is not None:
                painter.save()
                painter.resetTransform()
                solid = QtGui.QPen(pen)
                solid.setStyle(QtCore.Qt.SolidLine)
                solid.setJoinStyle(QtCore.Qt.MiterJoin)
                painter.setPen(solid)
                painter.setBrush(pen.color())
                painter.drawPolygon(head)
                painter.restore()

    def boundingRect(self):
        ''' with room for the arrowhead, which is sized in pixels '''
        rect = super().boundingRect()
        if self.arrow is None or self.scene() is None or not self.scene().views():
            return rect
        to_device = self.deviceTransform(self.scene().views()[0].viewportTransform())
        to_item, invertible = to_device.inverted()
        if not invertible:
            return rect
        reach = 2 * ARROW_LENGTH * max(self.pen().widthF(), 1.)
        dx = to_item.map(QtCore.QLineF(0, 0, reach, 0)).length()
        dy = to_item.map(QtCore.QLineF(0, 0, 0, reach)).length()
        return rect.adjusted(-dx, -dy, dx, dy)


def arrowhead(to_device, tip, start, width):
    '''
    the triangle (QPolygonF, device coordinates) of an arrowhead at view
    point tip, pointing from view point start, for a line of this pixel
    width; None if the two points are the same on screen
    '''
    tip = to_device.map(QtCore.QPointF(*tip))
    start = to_device.map(QtCore.QPointF(*start))
    d = np.array([tip.x() - start.x(), tip.y() - start.y()])
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return None
    d /= norm
    n = np.array([-d[1], d[0]])
    t = np.array([tip.x(), tip.y()])
    base = t - ARROW_LENGTH * width * d
    corners = [t, base + ARROW_HALF_WIDTH * width * n, base - ARROW_HALF_WIDTH * width * n]
    return QtGui.QPolygonF([QtCore.QPointF(*c) for c in corners])


def annotation_items(paths, name, line_style='solid', label_pos=None, on_moved=None, font_size=FONT_POINT_SIZE,
                     show_line=True, rotate=True, arrow=False):
    '''
    The graphics items annotating a particle in a view: its path (the
    segments it is drawn in, (n, 2) points each) in white, and its name above
    the longest segment. line_style: 'solid', 'dotted' (a straight line
    standing in for a trajectory not known: a neutral particle's, or one
    without steps) or 'dashed' (the incoming neutrino's). arrow: an
    arrowhead at the end of the last segment.
    label_pos: where the label was moved to (view coordinates), else its
    default place; on_moved(position): called when the user moves it.
    font_size: of the name, in points; show_line: False for the name only
    (still placed along the path); rotate: the name along the path, else
    level.
    '''
    paths = [np.asarray(p, dtype=float) for p in paths if len(p) > 1]
    if not paths:
        return []
    path = QtGui.QPainterPath()
    for points in paths:
        path.moveTo(*points[0])
        for x, y in points[1:]:
            path.lineTo(x, y)
    line = AnnotationPath(path, arrow=(tuple(paths[-1][-1]), tuple(paths[-1][max(0, len(paths[-1]) - 6)])) if arrow else None)
    pen = pg.mkPen(ANNOTATION_COLOR, width=LINE_WIDTH)
    pattern = LINE_PATTERNS[line_style]
    if pattern is not None:
        pen.setDashPattern(list(pattern))
    pen.setCosmetic(True)
    line.setPen(pen)
    line.setZValue(Z_VALUE)

    longest = max(paths, key=lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    (x, y), (dx, dy) = label_place(longest)
    # its bottom middle on the line, the text along the line in the view's axes (so through zooms), or level
    label = AnnotationLabel(name, color=ANNOTATION_COLOR, anchor=(0.5, 1.0), rotateAxis=(dx, dy) if rotate else None,
                            on_moved=on_moved)
    font = label.textItem.font()
    font.setBold(False)
    font.setPointSize(font_size)
    label.setFont(font)
    label.setPos(*(label_pos if label_pos is not None else (x, y)))
    label.setZValue(Z_VALUE + 1)
    return [line, label] if show_line else [label]


class AnnotationLabel(MovableLabel):
    '''
    A particle's name, which can be dragged (outlined on hover, as the event
    label); on_moved(position) is called with its view position once moved
    '''

    def __init__(self, *args, on_moved=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_moved = on_moved
        self._pressed_at = None

    def mousePressEvent(self, ev):
        self._pressed_at = self.pos()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self._on_moved is not None and self._pressed_at is not None and self.pos() != self._pressed_at:
            self._on_moved((self.pos().x(), self.pos().y()))
        self._pressed_at = None
