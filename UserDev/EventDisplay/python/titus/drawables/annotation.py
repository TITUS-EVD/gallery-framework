"""
Annotations of MC truth particles in the wire views (the MC truth dock's
Annotate boxes): the particle's projected path in solid white, over the
tracks (unless only the names are shown), and its name above the line,
along it.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from titus.gui.widgets import MovableLabel

ANNOTATION_COLOR = (255, 255, 255)
LINE_WIDTH = 2  # pixels
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


def annotation_items(paths, name, dashed=False, label_pos=None, on_moved=None, font_size=FONT_POINT_SIZE,
                     show_line=True):
    '''
    The graphics items annotating a particle in a view: its path (the
    segments it is drawn in, (n, 2) points each) in white, and its name above
    the longest segment. dashed: a straight line standing in for a
    trajectory not known (a neutral particle's, or one without steps).
    label_pos: where the label was moved to (view coordinates), else its
    default place; on_moved(position): called when the user moves it.
    font_size: of the name, in points; show_line: False for the name only
    (still placed along the path).
    '''
    paths = [np.asarray(p, dtype=float) for p in paths if len(p) > 1]
    if not paths:
        return []
    path = QtGui.QPainterPath()
    for points in paths:
        path.moveTo(*points[0])
        for x, y in points[1:]:
            path.lineTo(x, y)
    line = QtWidgets.QGraphicsPathItem(path)
    pen = pg.mkPen(ANNOTATION_COLOR, width=LINE_WIDTH, style=QtCore.Qt.DashLine if dashed else QtCore.Qt.SolidLine)
    pen.setCosmetic(True)
    line.setPen(pen)
    line.setZValue(Z_VALUE)

    longest = max(paths, key=lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    (x, y), (dx, dy) = label_place(longest)
    # its bottom middle on the line, the text along the line in the view's axes (so through zooms)
    label = AnnotationLabel(name, color=ANNOTATION_COLOR, anchor=(0.5, 1.0), rotateAxis=(dx, dy), on_moved=on_moved)
    font = label.textItem.font()
    font.setBold(True)
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
