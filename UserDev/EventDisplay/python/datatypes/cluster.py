from datatypes.database import recoBase
from pyqtgraph.Qt import QtGui, QtCore
from datatypes.connectedObjects import connectedBox, connectedCircle, boxCollection
from ROOT import evd, vector
import pyqtgraph as pg
from itertools import cycle


class cluster(recoBase):

    """docstring for cluster"""

    def __init__(self, geom):
        super(cluster, self).__init__()
        self._productName = 'cluster'
        self._process = evd.DrawCluster(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._n_planes = geom.nPlanes() * geom.nTPCs() * geom.nCryos()
        self.init()

        self._listOfClusters = []

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
    def drawObjects(self, view_manager):

        geom = view_manager._geometry

        for view in view_manager.getViewPorts():

            # Get the plane_id
            thisPlane = geom.getPlaneID(plane=view.plane(), tpc=0, cryo=view.cryostat())

            # Also get all the other planes
            additional_planes = geom.getOtherPlanes(plane_id=thisPlane)

            # extend the list of clusters
            for i in range(0, self._n_planes): self._listOfClusters.append([])

            clusters = self._process.getDataByPlane(thisPlane)
            self.drawClusterList(view, clusters, thisPlane, geom)

            # Draw the additional planes, if any
            for plane_id in additional_planes:
                clusters = self._process.getDataByPlane(plane_id)
                self.drawClusterList(view=view,
                                     clusters=clusters,
                                     thisPlane=plane_id,
                                     geom=geom,
                                     flip=geom.flipPlane(plane_id),
                                     shift=geom.shiftPlane(plane_id))



    def drawClusterList(self, view, clusters, thisPlane, geom, flip=False, shift=False):

            for cluster in clusters:
                # Now make the cluster
                cluster_box_coll = boxCollection()
                cluster_box_coll.setColor(next(self._clusterColors))
                cluster_box_coll.setPlane(thisPlane)

                # Keep track of the cluster for drawing management
                self._listOfClusters[thisPlane].append(cluster_box_coll)

                # draw the hits in this cluster:
                cluster_box_coll.drawHits(view, cluster, geom, flip, shift)


    def clearDrawnObjects(self, view_manager):
        geom = view_manager._geometry
        for view in view_manager.getViewPorts():
            # thisPlane = view.plane() + view.cryostat() * geom.nPlanes() * geom.nTPCs()
            thisPlane = geom.getPlaneID(plane=view.plane(), tpc=0, cryo=view.cryostat())
            for cluster in self._listOfClusters[thisPlane]:
                cluster.clearHits(view)

            for plane in geom.getOtherPlanes(plane_id=thisPlane):
                for cluster in self._listOfClusters[plane]:
                    cluster.clearHits(view)

        # clear the list:
        self._listOfClusters = []


    def getAutoRange(self, plane):
        w = self._process.getWireRange(plane)
        t = self._process.getTimeRange(plane)
        return [w.first, w.second], [t.first, t.second]

