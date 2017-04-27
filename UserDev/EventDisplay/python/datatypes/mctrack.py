from database import recoBase
from pyqtgraph.Qt import QtGui, QtCore
from ROOT import evd
import pyqtgraph as pg



from database import recoBase3D

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
                line = gl.GLLinePlotItem(pos=pts,color=(255,255,0,255))
                view.addItem(line)
                self._drawnObjects.append(line)

    
    # # Just be stupid and try to draw something:
    # cylinderPoints = gl.MeshData.cylinder(2,50,radius=[0,1],length=1)
    # cylinder = gl.GLMeshItem(meshdata=cylinderPoints,drawEdges=False,shader='shaded', glOptions='opaque')
    # cylinder.scale(10,10,10)
    # # cylinder.setGLOptions("additive")
    # self.addItem(cylinder)


except Exception, e:
    pass

