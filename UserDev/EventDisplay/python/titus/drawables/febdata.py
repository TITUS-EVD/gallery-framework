from titus.drawables import Drawable
from ROOT import evd


class FEBData(Drawable):
    def __init__(self, gallery_interface, geom):
        super().__init__(gallery_interface)
        self._process = evd.DrawFEBData(geom.getGeometryCore(),
                                        geom.getDetectorProperties())
        self._process.initialize()

    def getData(self):
        return self._process.getArray()
