import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from titus.display import ViewLayout, track_correction, track_segments, track_t0s
from titus.drawables.drawable import ReaderDrawable


class polyLine(QtWidgets.QGraphicsPathItem):
    def __init__(self, points, color=None, style=QtCore.Qt.SolidLine, *args):
        super(polyLine, self).__init__()
        self.setAcceptHoverEvents(True)
        self._points = points
        self._color = color if color is not None else (0, 0, 0)
        self._style = style

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

    def hoverEnterEvent(self, e):
        QtWidgets.QGraphicsPathItem.hoverEnterEvent(self, e)
        self.setPen(self._pen(5))
        self.update()

    def hoverLeaveEvent(self, e):
        QtWidgets.QGraphicsPathItem.hoverLeaveEvent(self, e)
        self.setPen(self._pen(2))
        self.update()


class Track(ReaderDrawable):
    '''
    recob::Track trajectories projected into the wire views
    (titus.display.projection). Drawn as stored, or corrected to lie on
    their hits, as the TPC module's "Correct 3D objects" option says:
    Pandora's T0 shift is undone and each point is drawn in its hits' TPC,
    using the T0s and track-hit associations in the file. Corrected tracks
    are dashed, and their tooltip says what was changed.
    '''
    KIND = 'tracks'
    COLOR = (130, 0, 0)

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'track'
        self._geom = geom
        self._module = tpc_module
        # per producer: what the corrections need, and the corrections once computed
        self._inputs = []
        self._corrections = []

    def on_data(self):
        self._inputs = [self._correction_inputs(tag, tracks) for tag, tracks in zip(self.producers(), self._data)]
        self._corrections = [{} for _ in self._data]

    def _correction_inputs(self, tag, tracks):
        ''' T0 per track and the track-hit associations, from the products related to these tracks '''
        tag = str(tag)
        track_hits = self.read_optional('track_hits', tag)
        hits = None
        if track_hits is not None and len(track_hits):
            hits = self.read_optional('hits', track_hits.product_tags[track_hits.right_product[0]])

        pfp_tracks = self.read_optional('pfp_tracks', tag)
        pfp_t0s = t0s = None
        if pfp_tracks is not None and len(pfp_tracks):
            pfp_tag = pfp_tracks.product_tags[pfp_tracks.left_product[0]]
            pfp_t0s = self.read_optional('pfp_t0s', pfp_tag)
            if pfp_t0s is not None and len(pfp_t0s):
                t0s = self.read_optional('t0s', pfp_t0s.product_tags[pfp_t0s.right_product[0]])
        return {'t0': track_t0s(len(tracks), pfp_tracks, pfp_t0s, t0s),
                'track_hits': track_hits if hits is not None else None, 'hits': hits}

    def _correction(self, producer, row, layout):
        cache = self._corrections[producer]
        if row not in cache:
            inputs = self._inputs[producer]
            cache[row] = track_correction(layout, self._data[producer], row, inputs['t0'][row],
                                          inputs['track_hits'], inputs['hits'])
        return cache[row]

    def corrected(self):
        return getattr(self._module, 'correct_3d_objects', lambda: True)()

    def drawObjects(self):
        corrected = self.corrected()
        for view in self._module._wire_views.values():
            layout = ViewLayout(self._geom, plane=view.plane(), cryostat=view.cryostat())
            for producer, tracks in enumerate(self._data):
                for row in range(len(tracks)):
                    correction = self._correction(producer, row, layout) if corrected else None
                    for segment in track_segments(layout, tracks, row, correction):
                        points = [QtCore.QPointF(x, y) for x, y in zip(segment.x, segment.y)]
                        moved = correction is not None and correction.corrected
                        line = polyLine(points, self.COLOR, QtCore.Qt.DashLine if moved else QtCore.Qt.SolidLine)
                        line.setToolTip(self.genToolTip(tracks, row, correction))
                        view._view.addItem(line)
                        self._drawnObjects.append(line)

    def genToolTip(self, tracks, row, correction):
        points = tracks.points(row)
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum() if len(points) > 1 else 0.
        what = correction.describe() if correction is not None else 'drawn as stored'
        return f'Track {tracks.id[row]}: length {length:.1f} cm, {len(points)} points\n{what}'
