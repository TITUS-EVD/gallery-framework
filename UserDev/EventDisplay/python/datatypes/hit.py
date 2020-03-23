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

    def __init__(self, geom):
        super(hit, self).__init__()
        self._productName = 'hit'
        self._process = evd.DrawHit(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._brush = (0, 0, 0)
        self._n_planes = geom.nPlanes() * geom.nTPCs() * geom.nCryos()
        self._n_tpcs = geom.nTPCs()
        self.init()

    def drawObjects(self, view_manager):

        geom = view_manager._geometry
        for view in view_manager.getViewPorts():
            thisPlane = view.plane() + view.cryostat() * geom.nPlanes() * geom.nTPCs()
            # print('view.plane()', view.plane(), 'view.cryostat()', view.cryostat(), 'geom.nPlanes()', geom.nPlanes(), 'thisPlane', thisPlane)
            for i in range(0, self._n_planes): self._drawnObjects.append([])
            # First get the hit information:
            hits = self._process.getDataByPlane(thisPlane)

            self.drawHitList(view, hits, thisPlane, geom)

            # In case of 2 TPCs, also draw the hits on
            # the other plane, but flipping the time
            if geom.nTPCs() == 2:
                for left_plane in geom.planeMix()[thisPlane]:
                    hits = self._process.getDataByPlane(left_plane)
                    self.drawHitList(view, hits, left_plane, geom)

    def drawHitList(self, view, hits, thisPlane, geom):
        if len(hits) == 0:
            return 
        for i in range(len(hits)):
            hit = hits[i]

            wire = hit.wire()
            time = hit.time() + geom.timeOffsetTicks(view.plane())
            width = 1
            height = hit.rms()

            location = hit.tpc()

            # Flip the time if odd tpc
            if hit.tpc() % 2 == 1:
                time = geom.tRange() - time

            # Shift up to the appropriate view
            time = time + location * geom.tRange()

            # Add the ad-hoc gap between TPCs
            time = time + location * geom.cathodeGap()

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
            time=hit.time(), 
            rms=hit.rms(), 
            integral=hit.charge())

    def getHitsOnWire(self, plane, wire):
        return self._process.getHitsOnWirePlane(wire,plane)
        