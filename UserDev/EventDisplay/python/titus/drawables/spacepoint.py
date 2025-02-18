from titus.drawables import Drawable
from ROOT import evd, TVector3
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import math as mt


class SpacePoint(Drawable):

    """docstring for spacepoint"""

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'spacepoint'
        self._process = evd.DrawSpacepoint(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._geom = geom
        self._module = tpc_module
        self.init()
    
    def genToolTip(self, SpacePoint):
        return 'Time: {time:0.1f}\nSpace Point ID: {ID}'.format(
            time=SpacePoint.time(),
            ID=SpacePoint.SpacePointID())

    def drawObjects(self):
        print("In Space point class drawObjects")

        for _, view in self._module._wire_views.items():
            print("In a view ")
            thisPlane = view.plane()
            self._drawnObjects.append([])
            spts = self._process.getDataByPlane(thisPlane)
            radBigW = 0.2 / self._geom.wire2cm()
            radBigT = (0.2) / self._geom.time2cm()
            offset = self._geom.offset(thisPlane) / self._geom.time2cm()
            print("Have ", len(spts), " space points")
            for i in range(len(spts)):
                thisPoint = spts[i]
                # Need to scale back into wire time coordinates:
                sW = thisPoint.wire() #/ self._geom.wire2cm()
                sT = thisPoint.time() #/ self._geom.time2cm() + offset
                print("Drawing space point at ", sW, " , " , sT)
                r = QtWidgets.QGraphicsEllipseItem(
                    sW -radBigW, sT-radBigT, 2*radBigW, 2*radBigT)
                r.setPen(pg.mkPen(255,0,255))
                # r.setBrush(pg.mkColor(255,0,255))
                # r.setBrush((0,0,0,opacity))
                r.setToolTip(self.genToolTip(thisPoint))
                self._drawnObjects[thisPlane].append(r)
                view._view.addItem(r)


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

    class spacepoint3D(recoBase3D):

        """docstring for spacepoint3D"""

        def __init__(self):
            super(spacepoint3D, self).__init__()
            self._productName = 'spacepoint3D'
            self._process = evd.DrawSpacepoint3D()
            self.init()

        def drawObjects(self, view_manager):

            geom = view_manager._geometry
            view = view_manager.getView()

            spts = self._process.getData()
            i_color = 0

            # Make a collection to add the points to:
            points = np.ndarray((spts.size(), 3))

            for i in range(len(spts)):
                thisPoint = spts[i]

                points[i][0] = thisPoint.X()
                points[i][1] = thisPoint.Y()
                points[i][2] = thisPoint.Z()

            glPointsCollection = gl.GLScatterPlotItem(pos=points, size=5)

            view.addItem(glPointsCollection)

            self._drawnObjects.append(glPointsCollection)


except:
    pass
