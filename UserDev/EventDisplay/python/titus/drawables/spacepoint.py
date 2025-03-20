from titus.drawables import Drawable
from ROOT import evd, TVector3
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import math as mt


class SpacePointGroup(QtWidgets.QGraphicsItemGroup):
    def __init__(self):
        super().__init__()
        self.setAcceptHoverEvents(True)
        self._isHighlighted = False
        self.EllipseList = []
        self.ObjectForHover = self
    def SetHoverParent(self, HoverParent):
        self.ObjectForHover = HoverParent
    def add_Ellipse(self, Ellipse): #Adds QGraphics item to group object
        self.addToGroup(Ellipse)
        self.EllipseList.append(Ellipse)
    def add_Subgroup(self, subgroup):
        self.addToGroup(subgroup)
        #need to combine the two rectangle lists
        self.EllipseList = self.EllipseList + subgroup.EllipseList
    #Direction fuction overrides 
    def hoverEnterEvent(self, e):
        self.ObjectForHover.Parent_hoverEnterEvent(e)
    def hoverLeaveEvent(self, e):
        self.ObjectForHover.Parent_hoverLeaveEvent(e)
    #Actual tools for the super gorup to use
    def Parent_hoverEnterEvent(self, e):
    #Change all the rectangles to yellow
        for i in range(0, len(self.EllipseList)):
            self.EllipseList[i].setPen(pg.mkPen(255,255,0))
            self.EllipseList[i].setBrush(pg.mkColor(255,255,0, 100))
        self.update()
    def Parent_hoverLeaveEvent(self, e):
        #Change the colors back
        for i in range(0, len(self.EllipseList)):
            self.EllipseList[i].setPen(pg.mkPen(255,0, 255))
            self.EllipseList[i].setBrush(pg.mkColor(255,0,255, 100))
        self.update()


class SpacePoint(Drawable):
    """docstring for spacepoint"""
    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'spacepoint'
        print("directory for geom objection in space point" , dir(geom))
        print('nCryos', 'nPlanes', 'nTPCs', 'nViews', 'name')
        print(geom.nCryos(), geom.nPlanes(), geom.nTPCs(), geom.nViews(), geom.name())
        print("directory for geom core", dir(geom.getGeometryCore()))
        print('nCryos', 'nPlanes', 'nTPCs', 'nViews', 'name')
        print(geom.getGeometryCore().Ncryostats(), geom.getGeometryCore().Nplanes(), geom.getGeometryCore().TotalNTPC(), 
              geom.getGeometryCore().Nviews(), geom.getGeometryCore().DetectorName())
        self._process = evd.DrawSpacepoint(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._geom = geom
        self._module = tpc_module
        self.init()
    def genToolTip(self, SpacePoint, ActualTimeForText, OriginatingPlane):
        return 'Time: {time:0.1f}\nSpace Point ID: {ID} \nPlaneBeingPlotted: {OriginatingPlane}'.format(
            time=ActualTimeForText,
            ID=SpacePoint.SpacePointID(), OriginatingPlane=OriginatingPlane)
    def drawObjects(self):
        #Annoying way to count the space points
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
            FullDetectorItemGroups.append(SpacePointGroup())
        for _, view in self._module._wire_views.items():
            thisPlane = view.plane()
            self._drawnObjects.append([])
            radBigW = 0.2 / self._geom.wire2cm()
            radBigT = (0.2) / self._geom.time2cm()
            CurrentPlanes = [thisPlane]
            for Plane in additional_planes:
                CurrentPlanes.append(Plane)
            TPCCounter=0
            for Plane in CurrentPlanes:
                spts = self._process.getDataByPlane(Plane)
                for i in range(len(spts)):
                    thisPoint = spts[i]
                    # Need to scale back into wire time coordinates:
                    sW = thisPoint.wire() / self._geom.wire2cm()
                    sT = thisPoint.time() / self._geom.time2cm()  + self._geom.timeOffsetTicks(thisPlane) 
                    if( TPCCounter == 1 ): 
                        # Flip the time
                        sT = self._geom.tRange() - sT
                        # Shift up to the appropriate view
                        sT = sT + self._geom.tRange()
                        # Add the ad-hoc gap between TPCs
                        sT = sT + self._geom.cathodeGap()-13 #Hardcoded offset to get blips to line up in west TPC. Talk to Marco about it
                    r = QtWidgets.QGraphicsEllipseItem(
                        sW -radBigW, sT-radBigT, 2*radBigW, 2*radBigT)
                    r.setPen(pg.mkPen(255,0,255))
                    r.setBrush(pg.mkColor(255,0,255, 100))
                    r.setToolTip(self.genToolTip(thisPoint, sT, Plane))
                    TempGroup = SpacePointGroup()
                    TempGroup.add_Ellipse(r)
                    # r.setBrush((0,0,0,opacity))
                    self._drawnObjects[thisPlane].append(TempGroup)
                    FullDetectorItemGroups[thisPoint.SpacePointID()].add_Subgroup(TempGroup) #Probably need to add full detector item groups to draw objects too
                    TempGroup.SetHoverParent(FullDetectorItemGroups[thisPoint.SpacePointID()])
                    view._view.addItem(TempGroup)
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