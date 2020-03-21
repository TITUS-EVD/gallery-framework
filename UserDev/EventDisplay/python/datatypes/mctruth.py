from datatypes.database import recoBase
from pyqtgraph.Qt import QtGui, QtCore
from ROOT import evd, larutil
import pyqtgraph as pg

pdg_to_name = {
    12: 'nue',
    -12: 'anue',
    14 : "numu",
    -14 : 'numubar'
}

mode_to_name = {
    0: 'QE',
    1: 'RES',
    2 : 'DIS',
    3 : 'COH',
    10: 'MEC'
}

class mctruth(recoBase):

    def __init__(self, geom):
        super(mctruth, self).__init__()
        self._productName = 'mctruth'
        self._process = evd.DrawMCTruth(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self.init()

    def getLabel(self):
        info = self._process.getData()
        # Determine type of incoming neutrino:
        inneut = {
        12 : "nue",
        -12 : "nuebar",
        14 : "numu",
        -14 : "numubar"
        }.get(info.incoming_lepton_pdg())

        # print info.target_pdg()

        # target = {
        # 2212 : "p",
        # 2112 : "n",
        # 1000180400 : "Ar"
        # }.get(info.target_pdg())

        return inneut + " + " 

    def drawObjects(self, view_manager):
        geom = view_manager._geometry

        # Just draw the vertex to start:
        mcts = self._process.getData()

        if len(mcts) == 0:
            return

        # Only the first neutrin ofor now
        mct = mcts[0]

        vertex = mct.vertex()

        for view in view_manager.getViewPorts():
            self._drawnObjects.append([])

            offset = geom.offset(view.plane())

            geo_helper = larutil.SimpleGeometryHelper(geom.getGeometryCore(), 
                                                      geom.getDetectorProperties())

            vertexPoint = geo_helper.Point_3Dto2D(vertex, view.plane(), 
                                                          view.tpc(), 
                                                          view.cryostat())
            
            points = self.makeCross(vertexPoint.w/geom.wire2cm(),
                                    (vertexPoint.t + offset )/geom.time2cm(),
                                    shortDistX=0.05/geom.wire2cm(),
                                    longDistX=1.0/geom.wire2cm(),
                                    shortDistY=0.05/geom.time2cm(),
                                    longDistY=1.0/geom.time2cm(),
                                    )

            thisPolyF = QtGui.QPolygonF(points)
            thisPoly = QtGui.QGraphicsPolygonItem(thisPolyF)
            thisPoly.setBrush(pg.mkColor((200,200,200,200)))

            thisPoly.setToolTip('Neutrino Interaction Vertex')

            self._drawnObjects[view.plane()].append(thisPoly)
            view._view.addItem(thisPoly)


        mb = view_manager.getMessageBar()
        message = str()
        tooltip = str()

        if mct.origin() == 1:
            pdg = pdg_to_name[mct.nu_pdg()]
            mode = mode_to_name[mct.int_mode()]
            message += f'PDG: {mct.nu_pdg()}, Neutrino Energy: {mct.nu_energy():.3} GeV, mode: {mode}'

            fs_pdgs = mct.finalstate_pdg()
            fs_enes = mct.finalstate_energy()
            print(fs_pdgs)
            tooltip += 'Final State Particles\n'
            for p, e in zip(fs_pdgs, fs_enes):
                tooltip += f'  PDG: {p}, Energy: {e:.3} GeV\n'

        elif mct.origin() == 2:
            message += f'Cosmic Origin'
        elif mct.origin() == 3:
            message += f'Supernovae Event'
        elif mct.origin() == 4:
            message += f'Single Particle Generation'


        mb.showMessage(message)
        mb.setToolTip(tooltip)

    def makeCross(self, startX, startY,
                  shortDistX, longDistX,
                  shortDistY, longDistY):
        # Let's draw an X to mark the vertex in each plane.
        points = []
        # Start by filling in points to mark the X:
        points.append(QtCore.QPointF(startX + shortDistX,
                                     startY))
        points.append(QtCore.QPointF(startX + shortDistX + longDistX,
                                     startY - longDistY))
        points.append(QtCore.QPointF(startX + longDistX,
                                     startY - shortDistY - longDistY))

        points.append(QtCore.QPointF(startX,
                                     startY - shortDistY))
        points.append(QtCore.QPointF(startX - longDistX,
                                     startY - longDistY - shortDistY))
        points.append(QtCore.QPointF(startX - longDistX - shortDistX,
                                     startY - longDistY))

        points.append(QtCore.QPointF(startX - shortDistX,
                                     startY))
        points.append(QtCore.QPointF(startX - shortDistX - longDistX,
                                     startY + longDistY))
        points.append(QtCore.QPointF(startX - longDistX,
                                     startY + shortDistY + longDistY))

        points.append(QtCore.QPointF(startX,
                                     startY + shortDistY))
        points.append(QtCore.QPointF(startX + longDistX,
                                     startY + shortDistY + longDistY))
        points.append(QtCore.QPointF(startX + longDistX + shortDistX,
                                     startY + longDistY))

        return points
