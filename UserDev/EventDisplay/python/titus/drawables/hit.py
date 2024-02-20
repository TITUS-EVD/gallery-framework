from titus.drawables import Drawable
from ROOT import evd
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
# from titus.modules import TpcModule

class hitBox(QtWidgets.QGraphicsRectItem):

    """docstring for hitBox"""

    def __init__(self, *args, **kwargs):
        super(hitBox, self).__init__(*args)
        self.setAcceptHoverEvents(True)
        self._isHighlighted = False

    def hoverEnterEvent(self, e):
        self.setPen(pg.mkPen('r', width=2))

    def hoverLeaveEvent(self, e):
        self.setPen(pg.mkPen(None))


class Hit(Drawable):

    """docstring for hit"""

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'hit'
        self._process = evd.DrawHit(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._brush = (0, 0, 0)
        self._n_planes = geom.nPlanes() * geom.nTPCs() * geom.nCryos()
        self._n_tpcs = geom.nTPCs()
        self._split_wire = geom.splitWire()
        self._geom = geom

        # this product is only drawn when TPC module is active, and it needs to
        # access the GUI elements in the TPC module class.
        self._module = tpc_module

        self.init()

    def drawObjects(self):
        # Loop over all the viewports
        for _, view in self._module._wire_views.items():
            # Get the plane_id
            thisPlane = self._geom.getPlaneID(plane=view.plane(), tpc=0, cryo=view.cryostat())

            # Also get all the other planes
            additional_planes = self._geom.getOtherPlanes(plane_id=thisPlane)

            for i in range(0, self._n_planes): self._drawnObjects.append([])

            # First get the hit information:
            hits = self._process.getDataByPlane(thisPlane)

            self.drawHitList(view, hits, thisPlane, self._geom)

            # Draw the additional planes, if any
            for plane_id in additional_planes:
                hits = self._process.getDataByPlane(plane_id)
                self.drawHitList(view=view,
                                 hits=hits,
                                 thisPlane=plane_id,
                                 geom=self._geom,
                                 flip=self._geom.flipPlane(plane_id),
                                 shift=self._geom.shiftPlane(plane_id))


    def drawHitList(self, view, hits, thisPlane, geom, flip=False, shift=False):
        if len(hits) == 0:
            return
        # for i in range(len(hits)):
        for i in range(100):
            hit = hits[i]

            wire = hit.wire()
            time = hit.time() + geom.timeOffsetTicks(view.plane())
            width = 1
            height = hit.rms()

            # Override for SBND commissioning fasthit finder
            if 'fasthit' in self._producer_name:
                height = hit.end_time() - hit.start_time() + 1

            if flip:
                # Flip the time
                time = geom.tRange() - time

                # Shift up to the appropriate view
                time = time + geom.tRange()

                # Add the ad-hoc gap between TPCs
                time = time + geom.cathodeGap()

            if shift:
                offset = (geom.wRange(view.plane()) - geom.cathodeGap()) / 2. + geom.cathodeGap()
                wire = wire + offset

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


    def clearDrawnObjects(self, obj_list=None):
        """ Override base class since our object list is nested """
        for view_objs in self._drawnObjects:
            for obj in view_objs:
                obj.scene().removeItem(obj)
        self._drawnObjects = []
