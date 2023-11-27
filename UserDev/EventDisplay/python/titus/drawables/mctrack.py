from titus.drawables import Drawable
from pyqtgraph.Qt import QtGui, QtCore
from ROOT import evd
import pyqtgraph as pg

from titus.drawables.track import polyLine


class MCTrack(Drawable):

    def __init__(self, gallery_interface, geom, tpc_module):
        super().__init__(gallery_interface)
        self._product_name = 'mctrack'
        self._process = evd.DrawMCTrack(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._geom = geom
        self._module = tpc_module
        self.init()

    def drawObjects(self, on_both_tpcs=False):

        for _, view in self._module._wire_views.items():
            self._drawnObjects.append([])
            tracks = self._process.getDataByPlane(view.plane())

            for i in range(len(tracks)):
                track = tracks[i]
                # construct a polygon for this track:
                points = []
                # Remeber - everything is in cm, but the display is in
                # wire/time!
                for i, pair in enumerate(track.track()):
                    x = pair.first / self._geom.wire2cm()
                    y = pair.second / self._geom.time2cm()

                    # If odd TPC, shift this piece of the track up
                    if track.tpc()[i] % 2:
                        y += 2 * self._geom.triggerOffset()
                        y += self._geom.cathodeGap()

                    points.append(QtCore.QPointF(x, y))

                if len(points) == 0:
                    continue

                #print ('MCTrack pdg', track.pdg())

                origin = track.origin()
                if (origin == 1): # neutrino origin
                    color = (128,128,128) # Gray
                else:
                    color = (0,0,0) # Black

                thisPoly = polyLine(points, color)

                time = track.time() - self._geom.getDetectorClocks().TriggerTime()
                thisPoly.setToolTip(f'PDG = {track.pdg()}\nTime = {time:.4} us\nEnergy = {track.energy()/1e3:.4} GeV\nProcess = {track.process()}')

                view._view.addItem(thisPoly)

                self._drawnObjects[view.plane()].append(thisPoly)

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
    class mctrack3D(recoBase3D):

        def __init__(self):
            super(mctrack3D, self).__init__()
            self._productName = 'mctrack3D'
            self._process = evd.DrawMCTrack3D()
            self.init()
            self._mesh = gl.MeshData()

        def toggleMCCosmic(self, toggleValue):
            self._process.SetViewCosmicOption(toggleValue)

        def drawObjects(self, view_manager):
            geom = view_manager._geometry
            view = view_manager.getView()


            tracks = self._process.getData()


            for track in tracks:

                # construct a line for this track:
                points = track.track()
                x = np.zeros(points.size())
                y = np.zeros(points.size())
                z = np.zeros(points.size())
                # x = numpy.ndarray()
                # x = numpy.ndarray()
                i = 0
                for point in points:
                    x[i] = point.X()
                    y[i] = point.Y()
                    z[i] = point.Z()
                    i+= 1

                pts = np.vstack([x,y,z]).transpose()
                pen = pg.mkPen((255,0,0), width=2)
                origin = track.origin()
                line = gl.GLLinePlotItem(pos=pts,color=(255,255,0,255))
                if (origin == 1): # neutrino origin
                    line = gl.GLLinePlotItem(pos=pts,color=(0, 176, 139,255))
                view.addItem(line)
                self._drawnObjects.append(line)


    # # Just be stupid and try to draw something:
    # cylinderPoints = gl.MeshData.cylinder(2,50,radius=[0,1],length=1)
    # cylinder = gl.GLMeshItem(meshdata=cylinderPoints,drawEdges=False,shader='shaded', glOptions='opaque')
    # cylinder.scale(10,10,10)
    # # cylinder.setGLOptions("additive")
    # self.addItem(cylinder)


except:
    pass

