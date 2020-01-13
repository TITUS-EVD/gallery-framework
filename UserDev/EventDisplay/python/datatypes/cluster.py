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
        self._process = evd.DrawCluster(geom.getGeometryCore(), geom.getDetectrorProperties())
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
            thisPlane = view.plane()

            # extend the list of clusters
            self._listOfClusters.append([])


            clusters = self._process.getDataByPlane(thisPlane)
            self.drawClusterList(view, clusters, thisPlane, geom)

            # In case of 2 TPCs, also draw the hits on
            # the other plane, but flipping the time
            if geom.nTPCs() == 2:
                if thisPlane == 0: left_plane = 4
                if thisPlane == 1: left_plane = 3
                if thisPlane == 2: left_plane = 5
                clusters_2 = self._process.getDataByPlane(left_plane)
                self.drawClusterList(view, clusters_2, thisPlane, geom, flip=True)


    def drawClusterList(self, view, clusters, thisPlane, geom, flip=False):
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
                cluster_box_coll.drawHits(view, cluster, geom, flip)

                colorIndex += 1
                if colorIndex >= len(self._clusterColors):
                    colorIndex = 0

    def clearDrawnObjects(self, view_manager):
        i_plane = 0
        # erase the clusters
        for plane in self._listOfClusters:
            view = view_manager.getViewPorts()[i_plane]
            i_plane += 1
            for cluster in plane:
                cluster.clearHits(view)


        self._listOfClusters = []

    def getAutoRange(self, plane):
        w = self._process.getWireRange(plane)
        t = self._process.getTimeRange(plane)
        return [w.first, w.second], [t.first, t.second]

