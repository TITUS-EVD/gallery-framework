from titus.drawables import Drawable
from pyqtgraph.Qt import QtGui, QtCore
from titus.drawables.connectedObjects import connectedBox, connectedCircle, boxCollection
from ROOT import evd, vector
import pyqtgraph as pg
from itertools import cycle


class Cluster(Drawable):

    """docstring for cluster"""

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'cluster'
        self._process = evd.DrawCluster(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._n_planes = geom.nPlanes() * geom.nTPCs() * geom.nCryos()
        self.init()
        self._geom = geom
        self._module = tpc_module

        # Defining the cluster colors:
        self._colors = [
            (0, 147, 147),  # dark teal
            (0, 0, 252),    # bright blue
            (156, 0, 156),  # purple
            (255, 0, 255),  # pink
            (255, 0, 0),    # red
            (175, 0, 0),    # red/brown
            (252, 127, 0),  # orange
            (102, 51, 0),   # brown
            (127, 127, 127),# dark gray
            (210, 210, 210),# gray
            (100, 253, 0)   # bright green
        ]
        self._clusterColors = cycle(self._colors)

    # this is the function that actually draws the cluster.
    def drawObjects(self):

        for _, view in self._module._wire_views.items():

            # Get the plane_id
            thisPlane = self._geom.getPlaneID(plane=view.plane(), tpc=0, cryo=view.cryostat())

            # Also get all the other planes
            additional_planes = self._geom.getOtherPlanes(plane_id=thisPlane)

            # extend the list of clusters
            for i in range(0, self._n_planes): self._drawnObjects.append([])

            clusters = self._process.getDataByPlane(thisPlane)
            self.drawClusterList(view, clusters, thisPlane, self._geom)

            # Draw the additional planes, if any
            for plane_id in additional_planes:
                clusters = self._process.getDataByPlane(plane_id)
                self.drawClusterList(view=view,
                                     clusters=clusters,
                                     thisPlane=plane_id,
                                     geom=self._geom,
                                     flip=self._geom.flipPlane(plane_id),
                                     shift=self._geom.shiftPlane(plane_id))



    def drawClusterList(self, view, clusters, thisPlane, geom, flip=False, shift=False):

            for cluster in clusters:
                # Now make the cluster
                cluster_box_coll = boxCollection()
                cluster_box_coll.setColor(next(self._clusterColors))
                cluster_box_coll.setPlane(thisPlane)

                # Keep track of the cluster for drawing management
                self._drawnObjects[thisPlane].append(cluster_box_coll)

                # draw the hits in this cluster:
                cluster_box_coll.drawHits(view, cluster, geom, flip, shift)
                break




    def getAutoRange(self, plane):
        w = self._process.getWireRange(plane)
        t = self._process.getTimeRange(plane)
        return [w.first, w.second], [t.first, t.second]


    def clearDrawnObjects(self, obj_list=None):
        """ Override base class since our object list is nested """
        for view_objs in self._drawnObjects:
            for obj in view_objs:
                obj.clearHits()
        self._drawnObjects = []
