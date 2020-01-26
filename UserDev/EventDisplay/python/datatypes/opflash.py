from datatypes.database import recoBase
from ROOT import evd, TVector3
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import math as mt
from ROOT import larutil

from datatypes.database import recoBase3D

class opflash(recoBase):
    def __init__(self, geom):
        super(opflash, self).__init__()
        self._productName = 'opflash'
        self._process = evd.DrawOpflash(geom.getGeometryCore(), geom.getDetectorProperties())
        self.init()

    def drawObjects(self, view_manager):
        geom = view_manager._geometry

        view = view_manager.getOpticalViewport()

        self._drawnObjects.append([])

        print ('Opflash drawing still needs implementation.')

        return

        for p in range(0, geom.nTPCs() * geom.nCryos()):
            flashes = self._process.getExtraData(p)
            print ('plane', p, 'len', len(flashes))
            view.setFlashesForPlane(p, flashes)







try:
    import pyqtgraph.opengl as gl
    import numpy as np

    class opflash3D(recoBase3D):

        """docstring for opflash3D"""

        triggerColor = (0, 1.0, 0, 0.2)
        preSpillColor = (1.0, 0, 0, 0.2)
        postSpillColor = (0, 0, 1.0, 0.2)

        def __init__(self):
            super(opflash3D, self).__init__()
            self._productName = 'opflash3D'
            self._process = evd.DrawOpflash3D()
            self.init()
            self._triggerOffset \
                = larutil.DetectorProperties.GetME().TriggerOffset()
            self._readOutWindowSize \
                = larutil.DetectorProperties.GetME().ReadOutWindowSize()

        def drawObjects(self, view_manager):

            geom = view_manager._geometry
            view = view_manager.getView()

            flashes = self._process.getData()
            i_color = 0

            # Each flash is drawn as an oval in Y/Z/X
            for i in range(len(flashes)):
                thisFlash = flashes[i]

                # Figure out what color this flash should be drawn as
                # Determine the offset to the trigger:

                # print thisFlash.time() - self._triggerOffset
                offset = 0.0
                if thisFlash.time() < 0:
                    offset = thisFlash.time(
                        ) / self._triggerOffset
                    color = [-offset*x + (1+offset)*y for x,y in zip(self.preSpillColor,self.triggerColor)]
                elif thisFlash > 0:
                    offset = thisFlash.time() / \
                        (self._readOutWindowSize - self._triggerOffset)
                    color = [offset*x + (1-offset)*y for x,y in zip(self.postSpillColor,self.triggerColor)]

                # print offset
                # print color

                flashSphere = gl.MeshData.sphere(15, 15, radius=1)
                flash = gl.GLMeshItem(meshdata=flashSphere,
                                      drawEdges=False,
                                      color=color,
                                      glOptions='translucent')
                # shader='shaded',
                # color=color,

                flash.translate(offset*100, thisFlash.y(), thisFlash.z())
                flash.scale(thisFlash.time_width(),
                            thisFlash.y_width(),
                            thisFlash.z_width())

                view.addItem(flash)

                self._drawnObjects.append(flash)


except:
    pass
