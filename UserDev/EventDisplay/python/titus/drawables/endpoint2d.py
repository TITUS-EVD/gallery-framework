from titus.drawables import Drawable
from ROOT import evd
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from ROOT import larutil


class EndPoint2D(Drawable):

    """docstring for endpoint2d"""

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'endpoint2d'
        self._process = evd.DrawEndpoint(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._brush = (0, 0, 0)
        self._geom = geom
        self._module = tpc_module
        self.init()

    # this is the function that actually draws the hits.
    def drawObjects(self):

        for _, view in self._module._wire_views.items():
            thisPlane = view.plane()
            self._drawnObjects.append([])
            # First get the hit information:
            endpoints = self._process.getDataByPlane(thisPlane)

            for i in range(len(endpoints)):
                point = endpoints[i]
                # Draws a circle at (x,y,radius = 0.5cm)
                radBigW = 0.5 / self._geom.wire2cm()
                radBigT = (0.5) / self._geom.time2cm()

                offset = self._geom.offset(
                    thisPlane) / self._geom.time2cm()

                sW = point.wire()
                sT = point.time()
                # sT = point.time() + offset

                r = QtGui.QGraphicsEllipseItem(
                    sW-radBigW, sT-radBigT, 2*radBigW, 2*radBigT)

                # r = QtGui.QGraphicsRectItem(
                #     point.wire(), point.time(), 1, point.strength())

                # opacity = point.charge() / self._process.maxCharge(thisPlane)
                opacity = 1
                # opacity = exp( 1 + hit.charge() / self._process.maxCharge(thisPlane))/exp(2);
                # # New Way
                # r.setPen(pg.mkPen(brush,width=2))
                # # r.setBrush(pg.mkColor((255,255,255)))

                # Old Way:
                r.setPen(pg.mkPen(None))
                r.setBrush(pg.mkColor(0))
                # r.setBrush((0,0,0,opacity))
                self._drawnObjects[thisPlane].append(r)
                view._view.addItem(r)

    def clearDrawnObjects(self, obj_list=None):
        """ Override base class since our object list is nested """
        for view_objs in self._drawnObjects:
            for obj in view_objs:
                obj.scene().removeItem(obj)
        self._drawnObjects = []
