from datatypes.database import recoBase
from pyqtgraph.Qt import QtGui, QtCore
from datatypes.connectedObjects import connectedBox, connectedCircle, boxCollection
from ROOT import evd, vector
import pyqtgraph as pg


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
        self._clusterColors = [
            (0, 147, 147),  # dark teal
            (0, 0, 252),   # bright blue
            (156, 0, 156),  # purple
            (255, 0, 255),  # pink
            (255, 0, 0),  # red
            (175, 0, 0),  # red/brown
            (252, 127, 0),  # orange
            (102, 51, 0),  # brown
            (127, 127, 127),  # dark gray
            (210, 210, 210),  # gray
            (100, 253, 0)  # bright green
        ]

    # this is the function that actually draws the cluster.
    def drawObjects(self, view_manager):

        geom = view_manager._geometry

        for view in view_manager.getViewPorts():
    
            # get the plane
            thisPlane = view.plane() + view.cryostat() * geom.nPlanes() * geom.nTPCs()

            # extend the list of clusters
            for i in range(0, self._n_planes): self._listOfClusters.append([])

            clusters = self._process.getDataByPlane(thisPlane)
            self.drawClusterList(view, clusters, thisPlane, geom)

            # In case of 2 TPCs, also draw the hits on
            # the other plane, but flipping the time
            if geom.nTPCs() == 2:
                for left_plane in geom.planeMix()[thisPlane]:
                    print('left_plane', left_plane)
                    clusters = self._process.getDataByPlane(left_plane)
                    self.drawClusterList(view, clusters, left_plane, geom)


    def drawClusterList(self, view, clusters, thisPlane, geom):
            colorIndex = 0
            for i in range(len(clusters)):
                cluster = clusters[i]
                # Now make the cluster
                cluster_box_coll = boxCollection()
                cluster_box_coll.setColor(self._clusterColors[colorIndex])
                cluster_box_coll.setPlane(thisPlane)

                # Keep track of the cluster for drawing management
                self._listOfClusters[thisPlane].append(cluster_box_coll)

                # draw the hits in this cluster:
                cluster_box_coll.drawHits(view, cluster, geom)

                colorIndex += 1
                if colorIndex >= len(self._clusterColors):
                    colorIndex = 0


    def clearDrawnObjects(self, view_manager):
        geom = view_manager._geometry
        for view in view_manager.getViewPorts():
            thisPlane = view.plane() + view.cryostat() * geom.nPlanes() * geom.nTPCs()
            for cluster in self._listOfClusters[thisPlane]:
                cluster.clearHits(view)
            for left_plane in view_manager._geometry.planeMix()[thisPlane]:
                for cluster in self._listOfClusters[left_plane]:
                    cluster.clearHits(view)

        # clear the list:
        self._listOfClusters = []


    def getAutoRange(self, plane):
        w = self._process.getWireRange(plane)
        t = self._process.getTimeRange(plane)
        return [w.first, w.second], [t.first, t.second]

