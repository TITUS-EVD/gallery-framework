from database import recoBase
from ROOT import evd
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from track import polyLine

class numuselection(recoBase):
    """docstring for numuselection"""
    def __init__(self):
        super(numuselection, self).__init__()
        self._productName = 'numuselection'
        self._process = evd.DrawNumuSelection()
        self._brush = (0, 0, 0)
        self.init()

         

    # this is the function that actually draws the numu selection output.
    def drawObjects(self, view_manager):

        geom = view_manager._geometry
        for view in view_manager.getViewPorts():
            thisPlane = view.plane()
            self._drawnObjects.append([])
            # First get the hit information:
            numus = self._process.getDataByPlane(thisPlane)

            for i in xrange(len(numus)):
                # Draw the vertex:

                vertex = numus[i].vertex()
                # Draws a circle at (x,y,radius = 0.5cm)
                radBigW = 3 / view_manager._geometry.wire2cm()
                radBigT = (3) / view_manager._geometry.time2cm()

                offset = view_manager._geometry.offset(
                    thisPlane) / view_manager._geometry.time2cm()

                sW = vertex.w / view_manager._geometry.wire2cm()
                sT = vertex.t / view_manager._geometry.time2cm() + offset

                r = QtGui.QGraphicsEllipseItem(
                    sW-radBigW, sT-radBigT, 2*radBigW, 2*radBigT)


                r.setPen(pg.mkPen(None))
                r.setBrush(pg.mkColor(139,0,139))
                self._drawnObjects[thisPlane].append(r)
                view._view.addItem(r)



                # Draw all the tracks:
                tracks = numus[i].tracks()
                for j in xrange(len(numus[i].tracks())):
                    track = tracks[j]
                    # construct a polygon for this track:
                    points = []
                    # Remeber - everything is in cm, but the display is in
                    # wire/time!
                    for pair in track.track():
                        x = pair.first / geom.wire2cm()
                        y = pair.second / geom.time2cm() + offset
                        points.append(QtCore.QPointF(x, y))

                    thisPoly = polyLine(points)

                    #Change the color here:
                    if j == numus[i].muon_index():
                        # Do something special with the muon
                        pen = pg.mkPen((238,130,238), width=2)
                    else:
                        pen = pg.mkPen((139,0,139), width=2)

                    thisPoly.setPen(pen)
                    # polyLine.draw(view._view)
                
                    view._view.addItem(thisPoly)

                    self._drawnObjects[view.plane()].append(thisPoly)
