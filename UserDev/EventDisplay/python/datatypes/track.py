from datatypes.database import recoBase
from pyqtgraph.Qt import QtGui, QtCore
from ROOT import evd
import pyqtgraph as pg


class polyLine(QtGui.QGraphicsPathItem):

    def __init__(self, points, pen=None):
        super(polyLine, self).__init__()
        self._points = points

        # Initialize a path:
        path = QtGui.QPainterPath()
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black)
        self.setPen(pen)

        # Fill the path:
        path.moveTo(points[0])
        for i in range(len(points)-1):
            path.lineTo(points[i+1])
        self.setPath(path)


class track(recoBase):

    def __init__(self, geom):
        super(track, self).__init__()
        self._productName = 'track'
        self._process = evd.DrawTrack(geom.getGeometryCore(), geom.getDetectorProperties())
        self._n_planes = geom.nPlanes() * geom.nTPCs() * geom.nCryos()
        self.init()

    def drawObjects(self, view_manager, on_both_tpcs=False):
        geom = view_manager._geometry

        for view in view_manager.getViewPorts():
            #   # get the showers from the process:
            # self._drawnObjects.append([])
            for i in range(0, self._n_planes): self._drawnObjects.append([])
            thisPlane = view.plane() + view.cryostat() * geom.nPlanes() * geom.nTPCs()

            tracks = self._process.getDataByPlane(thisPlane)
            offset = geom.offset(view.plane()) / geom.time2cm()

            # print ('Drawing tracks for plane', view.plane())
            self.drawTracks(view, tracks, offset, view.plane(), geom)

            if geom.nTPCs() == 2:
                for left_plane in geom.planeMix()[thisPlane]:
                    tracks = self._process.getDataByPlane(left_plane)
                    # print ('Drawing tracks for plane', left_plane)
                    self.drawTracks(view, tracks, offset, left_plane, geom) #, (255, 128, 0))


    def drawTracks(self, view, tracks, offset, plane, geom, color=(130,0,0)):

        if len(tracks) == 0:
            return

        # print ('  Cool. We have', len(tracks), ' tracks.')

        for i in range(len(tracks)):
            track = tracks[i]
            # construct a polygon for this track:
            points = []

            location = track.tpc()
            # print ('  location is', location)

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
  

                points.append(QtCore.QPointF(x, y))

            # self._drawnObjects[view.plane()].append(thisPoly)

            thisPoly = polyLine(points)
            pen = pg.mkPen(color, width=2)
            thisPoly.setPen(pen)
            # polyLine.draw(view._view)
            
            view._view.addItem(thisPoly)

            self._drawnObjects[plane].append(thisPoly)




from datatypes.database import recoBase3D

try:
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

