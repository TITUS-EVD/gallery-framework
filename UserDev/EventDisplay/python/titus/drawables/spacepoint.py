from titus.drawables import Drawable
from ROOT import evd, TVector3
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import math as mt
import copy


class SpacePointHitGroup(QtWidgets.QGraphicsItemGroup):
    def __init__(self):
        super().__init__()
        self.setAcceptHoverEvents(True)
        self._isHighlighted = False
        self.RectangleList = []
        self.ObjectForHover = self
    def SetHoverParent(self, HoverParent):
        self.ObjectForHover = HoverParent
    def add_hitbox(self, hitbox): #Adds QGraphics item to group object
        self.addToGroup(hitbox)
        self.RectangleList.append(hitbox)
    def add_Subgroup(self, subgroup):
        self.addToGroup(subgroup)
        #need to combine the two rectangle lists
        self.RectangleList = self.RectangleList + subgroup.RectangleList
    #Direction fuction overrides 
    def hoverEnterEvent(self, e):
        self.ObjectForHover.Parent_hoverEnterEvent(e)
    def hoverLeaveEvent(self, e):
        self.ObjectForHover.Parent_hoverLeaveEvent(e)
    #Actual tools for the super gorup to use
    def Parent_hoverEnterEvent(self, e):
    #Change all the rectangles to yellow
        for i in range(0, len(self.RectangleList)):
            self.RectangleList[i].setPen(pg.mkPen(255,255,0))
            self.RectangleList[i].setBrush(pg.mkColor(255,255,0, 100))
        self.update()
    def Parent_hoverLeaveEvent(self, e):
        #Change the colors back
        for i in range(0, len(self.RectangleList)):
            self.RectangleList[i].setPen(pg.mkPen(255,0, 255))
            self.RectangleList[i].setBrush(pg.mkColor(255,0,255, 100))
        self.update()

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
        #Do initialization of objects necessary for core loop over space point ID
        SpacePointIDs = []
        for _, view in self._module._wire_views.items():
            GrabbedPlane = view.plane()
            spts = self._process.getDataByPlane(GrabbedPlane)
            for i in range(0, len(spts)):
                SpacePointIDs.append(spts[i].SpacePointID())
            self._drawnObjects.append([])
            additional_planes = self._geom.getOtherPlanes(plane_id=GrabbedPlane)
            for Plane in additional_planes:
                self._drawnObjects.append([])
                spts = self._process.getDataByPlane(Plane)
                for i in range(0, len(spts)):
                    SpacePointIDs.append(spts[i].SpacePointID())
        TotalSpacePoints = len(set(SpacePointIDs)) #Used to initialize hits to form group holder
        FullDetectorItemGroups = []
        for i in range(0, TotalSpacePoints):
            FullDetectorItemGroups.append(SpacePointHitGroup())
        #Okay now we are ready to loop over the actual space points
        for _, view in self._module._wire_views.items():
            GrabbedPlane = view.plane()
            additional_planes = self._geom.getOtherPlanes(plane_id=GrabbedPlane)
            CurrentPlanes = [GrabbedPlane]
            for Plane in additional_planes:
                CurrentPlanes.append(Plane)
            TPCCounter=0
            for thisPlane in CurrentPlanes:
                spts = self._process.getDataByPlane(thisPlane)
                SpacePointTrackerID = -1
                HitsToFormGroup = []
                for i in range(len(spts)):
                    thisPoint = spts[i]
                    CurrentID = thisPoint.SpacePointID()
                    if(SpacePointTrackerID!=CurrentID):
                    #    #Handle first entry 
                        if(SpacePointTrackerID==-1):
                            #Just set tracker
                            SpacePointTrackerID=CurrentID
                        else:
                            #We should check existing groups for any matching space point ID
                            #Add a new group to the scene
                            NewGroup = SpacePointHitGroup()
                            for Hit in HitsToFormGroup:
                                NewGroup.add_hitbox(Hit)
                            NewGroup.setToolTip(self.genToolTip(prevPoint))
                            NewGroup.SetHoverParent(FullDetectorItemGroups[SpacePointTrackerID])
                            FullDetectorItemGroups[SpacePointTrackerID].add_Subgroup(NewGroup)
                            HitsToFormGroup=[]
                            SpacePointTrackerID=CurrentID
                            self._drawnObjects[thisPlane].append(NewGroup)
                            view._view.addItem(NewGroup)
                    #Prepare rectangles for each hit associated with a space point
                    # Need to scale back into wire time coordinates:
                    sW = thisPoint.wire()
                    sT = thisPoint.time() + self._geom.timeOffsetTicks(thisPlane)
                    #Equivalent to the flip command in connectedObjects.py boxCollection.DrawHits
                    if(TPCCounter==1):
                        # Flip the time
                        sT = self._geom.tRange() - sT
                        # Shift up to the appropriate view
                        sT = sT + self._geom.tRange()
                        # Add the ad-hoc gap between TPCs
                        sT = sT + self._geom.cathodeGap()-13 #Hardcoded offset to get blips to line up in west TPC. Talk to Marco about it
                    width = 1
                    height = thisPoint.duration()
                    r = QtWidgets.QGraphicsRectItem(sW,
                        sT,
                        width,
                        height)
                    r.setPen(pg.mkPen(255,0,255))
                    r.setBrush(pg.mkColor(255,0,255, 100))
                    prevPoint = copy.copy(thisPoint)
                    HitsToFormGroup.append(r)
                    if(i==len(spts)-1): #Last entry do same as above
                        NewGroup = SpacePointHitGroup()
                        for Hit in HitsToFormGroup:
                            NewGroup.add_hitbox(Hit)
                        NewGroup.setToolTip(self.genToolTip(prevPoint))
                        FullDetectorItemGroups[SpacePointTrackerID].setToolTip(self.genToolTip(prevPoint)) #Should really do this at instantiation
                        FullDetectorItemGroups[SpacePointTrackerID].add_Subgroup(NewGroup)
                        HitsToFormGroup=[]
                        SpacePointTrackerID=CurrentID
                        self._drawnObjects[thisPlane].append(NewGroup)
                        view._view.addItem(NewGroup)
                TPCCounter=TPCCounter+1
        

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
