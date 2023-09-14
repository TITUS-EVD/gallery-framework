from datatypes.database import dataBase
from ROOT import evd


class febdata(dataBase):
    def __init__(self, geom):
        super(febdata, self).__init__()
        self._process = evd.DrawFEBData(geom.getGeometryCore(),
                                        geom.getDetectorProperties(),
                                        geom.getDetectorClocks())
        self._process.initialize()

    def getData(self):
        return self._process.getArray()
