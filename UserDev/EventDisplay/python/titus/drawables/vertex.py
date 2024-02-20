from titus.drawables import Drawable
from ROOT import evd
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg


class Vertex(Drawable):

    """docstring for vertex"""

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'vertex'
        self._process = evd.DrawVertex(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._brush = (0, 0, 0)
        self._geom = geom
        self._module = tpc_module
        self.init()

    # this is the function that actually draws the hits.
    def drawObjects(self):

        for _, view in self._module._wire_views.items():
            thisPlane = view.plane()
            self._drawnObjects.append([])
            # First get the vertex information:
            vertexes = self._process.getDataByPlane(thisPlane)

            for i in range(len(vertexes)):
                point = vertexes[i]
                # Draws a circle at (x,y,radius = 0.5cm)
                radBigW = 0.5 / self._geom.wire2cm()
                radBigT = (0.5) / self._geom.time2cm()

                offset = self._geom.offset(thisPlane) / self._geom.time2cm()

                sW = point.w / self._geom.wire2cm()
                sT = point.t / self._geom.time2cm() + offset

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
                r.setBrush(pg.mkColor(0,255,255))
                # r.setBrush((0,0,0,opacity))
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

    class vertex3D(recoBase3D):

        """docstring for vertex3D"""

        def __init__(self):
            super(vertex3D, self).__init__()
            self._productName = 'vertex3D'
            self._process = evd.DrawVertex3D()
            self.init()

        def drawObjects(self, view_manager):

            geom = view_manager._geometry
            view = view_manager.getView()

            vertexes = self._process.getData()

            for i in range(len(vertexes)):
                vertex = vertexes[i]

                # Stupid thing right now:
                # make a 3 lines around the vertex

                xline = np.ndarray((2, 3))
                yline = np.ndarray((2, 3))
                zline = np.ndarray((2, 3))

                length = 5

                for line in xline, yline, zline:
                    for point in line:
                        point[0] = vertex.X()
                        point[1] = vertex.Y()
                        point[2] = vertex.Z()

                xline[0][0] += length/2
                xline[1][0] -= length/2
                yline[0][1] += length/2
                yline[1][1] -= length/2
                zline[0][2] += length/2
                zline[1][2] -= length/2

                # Make the 3 lines for the vertex:

                # pts = np.vstack([x, y, z]).transpose()
                # pen = pg.mkPen((255, 0, 0), width=2)
                xglline = gl.GLLinePlotItem(pos=xline, width=3,
                                            color=(0.6, 0.51, 1.0, 1.0))
                yglline = gl.GLLinePlotItem(pos=yline, width=3,
                                            color=(0.6, 0.51, 1.0, 1.0))
                zglline = gl.GLLinePlotItem(pos=zline, width=3,
                                            color=(0.6, 0.51, 1.0, 1.0))
                view.addItem(xglline)
                view.addItem(yglline)
                view.addItem(zglline)
                self._drawnObjects.append(xglline)
                self._drawnObjects.append(yglline)
                self._drawnObjects.append(zglline)

except:
    pass
