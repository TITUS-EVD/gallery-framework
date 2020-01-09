from datatypes.database import recoBase
from ROOT import evd
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

class hitBox(QtGui.QGraphicsRectItem):

    """docstring for hitBox"""

    def __init__(self, *args, **kwargs):
        super(hitBox, self).__init__(*args)
        self.setAcceptHoverEvents(True)
        self._isHighlighted = False

    def hoverEnterEvent(self, e):
        self.setPen(pg.mkPen('r', width=2))

    def hoverLeaveEvent(self, e):
        self.setPen(pg.mkPen(None))


class hit(recoBase):

    """docstring for hit"""

    def __init__(self):
        super(hit, self).__init__()
        self._productName = 'hit'
        self._process = evd.DrawHit()
        self._brush = (0, 0, 0)
        self.init()

    def drawObjects(self, view_manager):

        geom = view_manager._geometry
        for view in view_manager.getViewPorts():
            thisPlane = view.plane()
            self._drawnObjects.append([])
            # First get the hit information:
            hits = self._process.getDataByPlane(thisPlane)

            self.drawHitList(view, hits, thisPlane, geom)

            # In case of 2 TPCs, also draw the hits on
            # the other plane, but flipping the time
            if geom.nTPCs() == 2:
                if thisPlane == 0: left_plane = 4
                if thisPlane == 1: left_plane = 3
                if thisPlane == 2: left_plane = 5
                hits_2 = self._process.getDataByPlane(left_plane)
                self.drawHitList(view, hits_2, thisPlane, geom, flip=True)

    def drawHitList(self, view, hits, thisPlane, geom, flip=False):
        for i in range(len(hits)):
            hit = hits[i]

            wire = hit.wire()
            time = hit.time() + geom.timeOffsetTicks(view.plane())
            width = 1
            height = hit.rms()

            if flip:
                time = 2 * geom.tRange() - time + geom.cathodeGap()

            # Draws a rectangle at (x,y,xlength, ylength)
            # r = QtGui.QGraphicsRectItem(wire, 
            #                             time, 
            #                             width, 
            #                             height)

            r = hitBox(wire, 
                       time, 
                       width, 
                       height)

            opacity = int(128 * hit.charge() / self._process.maxCharge(thisPlane)) + 127
            # opacity = exp( 1 + hit.charge() / self._process.maxCharge(thisPlane))/exp(2);
            # # New Way
            # r.setPen(pg.mkPen(brush,width=2))
            # # r.setBrush(pg.mkColor((255,255,255)))

            # Old Way:
            r.setPen(pg.mkPen(None))
            r.setBrush(pg.mkColor(0,0,0,opacity))
            # r.setBrush((0,0,0,opacity))

            r.setToolTip(self.genToolTip(hit))
           
            self._drawnObjects[thisPlane].append(r)
            view._view.addItem(r)

    def genToolTip(self, hit):
        return 'Time: {time:0.1f}\nRMS: {rms:0.1f}\nIntegral: {integral:0.1f}'.format(
            time=hit.wire(), 
            rms=hit.rms(), 
            integral=hit.charge())

    def getHitsOnWire(self, plane, wire):
        return self._process.getHitsOnWirePlane(wire,plane)
        