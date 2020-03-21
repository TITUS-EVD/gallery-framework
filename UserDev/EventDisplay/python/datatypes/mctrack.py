from datatypes.database import recoBase
from pyqtgraph.Qt import QtGui, QtCore
from ROOT import evd
import pyqtgraph as pg

from datatypes.track import polyLine


class mctrack(recoBase):

    def __init__(self, geom):
        super(mctrack, self).__init__()
        self._productName = 'mctrack'
        self._process = evd.DrawMCTrack(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self.init()

    def drawObjects(self, view_manager, on_both_tpcs=False):
        geom = view_manager._geometry

        for view in view_manager.getViewPorts():
            #   # get the showers from the process:
            self._drawnObjects.append([])
            tracks = self._process.getDataByPlane(view.plane())
            offset = geom.offset(view.plane()) / geom.time2cm()

            for i in range(len(tracks)):
                track = tracks[i]
                # construct a polygon for this track:
                points = []
                # Remeber - everything is in cm, but the display is in
                # wire/time!
                for pair in track.track():
                    x = pair.first / geom.wire2cm()
                    y = pair.second / geom.time2cm() + offset
                    if geom.nTPCs() == 2 and on_both_tpcs:
                        cathode_time = (2 * geom.halfwidth() + geom.offset(view.plane()))/geom.time2cm()
                        if y > cathode_time:
                            y += geom.tRange() - geom.triggerOffset()
                    points.append(QtCore.QPointF(x, y))

                # self._drawnObjects[view.plane()].append(thisPoly)

                if len(points) == 0:
                    continue

                

                #print ('MCTrack pdg', track.pdg())

                origin = track.origin()
                if (origin == 1): # neutrino origin
                    color = (128,128,128) # Gray
                else:
                    color = (0,0,0) # Black

                thisPoly = polyLine(points, color)

                time = track.time()
                thisPoly.setToolTip(f'PDG = {track.pdg()}\nTime = {track.time():.4} us\nEnergy = {track.energy()/1e3:.4} GeV\nProcess = {track.process()}')

                view._view.addItem(thisPoly)

                self._drawnObjects[view.plane()].append(thisPoly)


from datatypes.database import recoBase3D

try:
    import pyqtgraph.opengl as gl
    import numpy as np
    class mctrack3D(recoBase3D):

        def __init__(self):
            super(mctrack3D, self).__init__()
            self._productName = 'mctrack3D'
            self._process = evd.DrawMCTrack3D()
            self.init()
            self._mesh = gl.MeshData()

        def toggleMCCosmic(self, toggleValue):
            self._process.SetViewCosmicOption(toggleValue)

        def drawObjects(self, view_manager):
            geom = view_manager._geometry
            view = view_manager.getView()


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
                origin = track.origin()
                line = gl.GLLinePlotItem(pos=pts,color=(255,255,0,255))
                if (origin == 1): # neutrino origin
                    line = gl.GLLinePlotItem(pos=pts,color=(0, 176, 139,255))
                view.addItem(line)
                self._drawnObjects.append(line)

    
    # # Just be stupid and try to draw something:
    # cylinderPoints = gl.MeshData.cylinder(2,50,radius=[0,1],length=1)
    # cylinder = gl.GLMeshItem(meshdata=cylinderPoints,drawEdges=False,shader='shaded', glOptions='opaque')
    # cylinder.scale(10,10,10)
    # # cylinder.setGLOptions("additive")
    # self.addItem(cylinder)


except:
    pass

