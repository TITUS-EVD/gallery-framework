from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

class HitBoxCollection(QtWidgets.QGraphicsItem):
    '''
    A collection of HitBox classes to be drawn at the same time.
    '''

    def __init__(self):
        super().__init__()
        self._rects = []

    def __len__(self):
        return len(self._rects)

    def add_hitboxes(self, rects):
        self._rects = rects

    def setPen(self, *args, **kwargs):
        for rect in self._rects:
            rect.setPen(*args, **kwargs)

    def __len__(self):
        return len(self._rects)

    def boundingRect(self):
        # Calculate the bounding rectangle of the entire group of rects TODO
        return QtCore.QRectF(0, 0, 200, 200)
    
    def paint(self, painter, option, widget):
        # This method is intentionally left empty because QGraphicsRectItem
        # will handle their own painting.
        pass


# This class wraps the hit object to allow them to all function together
class connectedBox(QtWidgets.QGraphicsRectItem):

    """docstring for connectedBox"""

    def __init__(self, *args, **kwargs):
        super(connectedBox, self).__init__(*args)
        self.setAcceptHoverEvents(True)
        self._isHighlighted = False

    def hoverEnterEvent(self, e):
        self.setToolTip(self._ownerToolTip())
        self._ownerHoverEnter(e)

    def hoverLeaveEvent(self, e):
        self._ownerHoverExit(e)

    def mouseDoubleClickEvent(self, e):
        self._toggleHighlight()

    def connectOwnerHoverEnter(self, ownerHoverEnter):
        self._ownerHoverEnter = ownerHoverEnter

    def connectOwnerHoverExit(self, ownerHoverExit):
        self._ownerHoverExit = ownerHoverExit

    def connectToggleHighlight(self, ownerTH):
        self._toggleHighlight = ownerTH

    def connectToolTip(self, ownerToolTip):
        self._ownerToolTip = ownerToolTip


class connectedCircle(QtWidgets.QGraphicsEllipseItem):

    """docstring for connectedCircle"""

    def __init__(self, *args, **kwargs):
        super(connectedCircle, self).__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self._isHighlighted = False

    def hoverEnterEvent(self, e):
        self.setToolTip(self._ownerToolTip())
        self._ownerHoverEnter(e)

    def hoverLeaveEvent(self, e):
        self._ownerHoverExit(e)

    def mouseDoubleClickEvent(self, e):
        self._toggleHighlight()

    def connectOwnerHoverEnter(self, ownerHoverEnter):
        self._ownerHoverEnter = ownerHoverEnter

    def connectOwnerHoverExit(self, ownerHoverExit):
        self._ownerHoverExit = ownerHoverExit

    def connectToggleHighlight(self, ownerTH):
        self._toggleHighlight = ownerTH

    def connectToolTip(self, ownerToolTip):
        self._ownerToolTip = ownerToolTip


class boxCollection(QtCore.QObject):
    # This class wraps a collection of hits and connects them together
    # it can draw and delete itself when provided with view_manage
    #
    # Provide some signals to communicate with cluster params
    mouseEnter = QtCore.pyqtSignal(QtWidgets.QGraphicsSceneHoverEvent)
    mouseExit = QtCore.pyqtSignal(QtWidgets.QGraphicsSceneHoverEvent)
    highlightChange = QtCore.pyqtSignal()

    def __init__(self):
        super(boxCollection, self).__init__()
        self._color = (0, 0, 0)
        self._plane = -1
        self._hitCollection = None
        self._isHighlighted = False
        self._params = None
        self._acceptHoverEvents = False

    def setColor(self, color):
        self._color = color

    def setPlane(self, plane):
        self._plane = plane

    def attachParams(self, params):
        self._params = params

    # Can connect boxCollections to other boxCollections or to cluster params
    def connect(self, other):
        self.mouseEnter.connect(other.hoverEnter)
        self.mouseExit.connect(other.hoverExit)
        self.highlightChange.connect(other.toggleHighlight)

    def genToolTip(self):
        if self._params == None:
            nhits = len(self._hitCollection)
            tip = "Hits: " + str(nhits)
            return tip
        else:
            return self._params.toolTip()

    def hoverEnter(self, e):
        self._hitCollection.setPen(pg.mkPen((0, 0, 0), width=1))
        # When the function is called from a box, the sender is none
        # When its passed from the params, the sender is something
        if self.sender() == None:
            self.mouseEnter.emit(e)

    def hoverExit(self, e):
        if self._isHighlighted:
            return
        self._hitCollection.setPen(pg.mkPen(None))
        # When the function is called from a box, the sender is none
        # When its passed from the params, the sender is something
        if self.sender() == None:
            self.mouseExit.emit(e)

    def toggleHighlight(self):
        self._isHighlighted = not self._isHighlighted
        # When the function is called from a box, the sender is none
        # When its passed from the params, the sender is something
        if self.sender() == None:
            self.highlightChange.emit()

    def drawHits(self, view, cluster, geom, flip=False, shift=False):

        hit_collection = HitBoxCollection()
        rects = []

        for i in range(len(cluster)):
            hit = cluster[i]

            wire = hit.wire()
            time = hit.time() + geom.timeOffsetTicks(view.plane())
            width = 1
            height = hit.rms()

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

            # Draws a rectangle at (x,y,xlength, ylength)
            r = connectedBox(wire, time, width, height, hit_collection)
            r.setPen(pg.mkPen(None))
            r.setBrush(pg.mkColor(self._color))
            # self._hitCollection.append(r)
            # view._view.addItem(r)

            # Connect the hit's actions with the clusters functions
            r.connectOwnerHoverEnter(self.hoverEnter)
            r.connectOwnerHoverExit(self.hoverExit)
            r.connectToggleHighlight(self.toggleHighlight)
            r.connectToolTip(self.genToolTip)

            rects.append(r)

        hit_collection.add_hitboxes(rects)
        self._hitCollection = hit_collection
        view._view.addItem(hit_collection)

    def clearHits(self):
        self._hitCollection.scene().removeItem(self._hitCollection)
        self._hitCollection = None
