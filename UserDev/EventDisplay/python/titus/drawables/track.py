from titus.drawables import Drawable
from PyQt5 import QtWidgets, QtGui, QtCore
from ROOT import evd
import pyqtgraph as pg


class polyLine(QtWidgets.QGraphicsPathItem):
    def __init__(self, points, color=None, *args):
        super(polyLine, self).__init__()
        self.setAcceptHoverEvents(True)
        self._points = points
        self._color = color

        # Initialize a path:
        path = QtGui.QPainterPath()
        if self._color is None:
            pen = QtGui.QPen(QtCore.Qt.black)
            self._color = (0, 0, 0)
        else:
            pen = pg.mkPen(color, width=2)
        self.setPen(pen)

        # Fill the path:
        path.moveTo(points[0])
        for i in range(len(points)-1):
            path.lineTo(points[i+1])
        self.setPath(path)


    def hoverEnterEvent(self, e):
        QtWidgets.QGraphicsPathItem.hoverEnterEvent(self, e)
        self.setPen(pg.mkPen(self._color, width=5))
        self.update()

    def hoverLeaveEvent(self, e):
        QtWidgets.QGraphicsPathItem.hoverLeaveEvent(self, e)
        self.setPen(pg.mkPen(self._color, width=2))
        self.update()


class Track(Drawable):
    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'track'
        self._process = evd.DrawTrack(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._process._projections_match = geom.projectionsMatch()
        print('self._process._projections_match?', self._process._projections_match)
        self._n_planes = geom.nPlanes() * geom.nTPCs() * geom.nCryos()
        self._geom = geom
        self._module = tpc_module
        self.init()


    def drawObjects(self, on_both_tpcs=False):
        for _, view in self._module._wire_views.items():
            self._drawnObjects.append([])
            tracks = self._process.getDataByPlane(view.plane())

            plane = view.plane()

            for i in range(len(tracks)):
                track = tracks[i]

                # construct a polygon for this track:
                points = []
                # Remeber - everything is in cm, but the display is in
                # wire/time!
                for i, pair in enumerate(track.track()):
                    x = pair.first / self._geom.wire2cm()
                    y = pair.second / self._geom.time2cm()

                    if track.tpc()[i] == 1:
                        # flip
                        y = self._geom.tRange() - y
                        # move up
                        y += self._geom.tRange()
                        # add cathode gap
                        y += self._geom.cathodeGap()

                    points.append(QtCore.QPointF(x, y))

                if len(points) == 0:
                    continue

                color = (130,0,0) # red

                thisPoly = polyLine(points, color)

                thisPoly.setToolTip(f'Length: {track.length():.2f} cm;  Theta: {track.theta():.2f};  Phi: {track.phi():.2f}')

                view._view.addItem(thisPoly)

                self._drawnObjects[view.plane()].append(thisPoly)


    def drawTracks(self, view, tracks, offset, plane, geom, color=(130,0,0)):

        if len(tracks) == 0:
            return

        print ('  Cool. We have', len(tracks), ' tracks.')

        for i in range(len(tracks)):
            track = tracks[i]
            # construct a polygon for this track:
            points = []

            location = track.tpc()

            # Remeber - everything is in cm, but the display is in
            # wire/time!
            for pair in track.track():
                x = pair.first / geom.wire2cm()
                y = pair.second / geom.time2cm() + offset

                # cathode_x = geom.getGeometryCore().Plane(0, track.tpc(), track.cryo()).GetCenter().X() - 2 * geom.getGeometryCore().DetHalfWidth()
                # print ('cathode_x', cathode_x)
                # y -= cathode_x

                plane_x = geom.getGeometryCore().Plane(view.plane(), track.tpc(), track.cryo()).GetCenter().X()
                plane_x_ref = geom.getGeometryCore().Plane(0, 0, 0).GetCenter().X()

                y -= location * (2 * geom.halfwidth()) / geom.time2cm()
                y += location * (plane_x - plane_x_ref - 4 * geom.halfwidth()) / geom.time2cm()
                y += location * (geom.tRange() + geom.cathodeGap())
                y -= location * 185


                points.append(QtCore.QPointF(x, y))

            # self._drawnObjects[view.plane()].append(thisPoly)

            thisPoly = polyLine(points, color)
            # pen = pg.mkPen(color, width=2)
            # thisPoly.setPen(pen)
            # polyLine.draw(view._view)

            thisPoly.setToolTip('Temp')

            view._view.addItem(thisPoly)

            self._drawnObjects[plane].append(thisPoly)

    def clearDrawnObjects(self, obj_list=None):
        """ Override base class since our object list is nested """
        for view_objs in self._drawnObjects:
            for obj in view_objs:
                obj.scene().removeItem(obj)
        self._drawnObjects = []




try:
    from gallery_interface.datatypes.database import recoBase3D
    import pyqtgraph.opengl as gl
    import numpy as np
    class track3D(recoBase3D):

        def __init__(self):
            super(track3D, self).__init__()
            self._productName = 'track3D'
            self._process = evd.DrawTrack3D(geom.getGeometryCore(), geom.getDetectrorProperties())
            self.init()


        def drawObjects(self, view_manager):
            geom = view_manager._geometry
            view = view_manager.getView()

            self
            tracks = self._process.getData()

            for track in tracks:

                # construct a line for this track:
                points = track.track()
                x = np.zeros(points.size())
                y = np.zeros(points.size())
                z = np.zeros(points.size())
                # x = numpy.ndarray()
                # x = numpy.ndarray()
                i = 0
                for point in points:
                    x[i] = point.X()
                    y[i] = point.Y()
                    z[i] = point.Z()
                    i+= 1

                pts = np.vstack([x,y,z]).transpose()
                pen = pg.mkPen((255,0,0), width=2)
                line = gl.GLLinePlotItem(pos=pts,color=(255,0,0,255), width=4)
                view.addItem(line)
                self._drawnObjects.append(line)



except:
    pass

