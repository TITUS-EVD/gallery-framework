from titus.drawables import Drawable
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

class MCTruth(Drawable):

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'mctruth'
        self._process = evd.DrawMCTruth(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._geom = geom
        self._module = tpc_module
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

    def drawObjects(self):

        # Just draw the vertex to start:
        mcts = self._process.getData()

        if len(mcts) == 0:
            return

        # Only the first neutrino for now
        mct = mcts[0]

        vertex = mct.vertex()

        for _, view in self._module._wire_views.items():
            self._drawnObjects.append([])

            geo_helper = larutil.SimpleGeometryHelper(self._geom.getGeometryCore(),
                                                      self._geom.getDetectorProperties(),
                                                      self._geom.getDetectorClocks())

            vertexPoint = geo_helper.Point_3Dto2D(vertex, view.plane())

            points = self.makeCross(startX=vertexPoint.w/self._geom.wire2cm(),
                                    startY=vertexPoint.t/self._geom.time2cm(),
                                    shortDistX=0.05/self._geom.wire2cm(),
                                    longDistX=1.0/self._geom.wire2cm(),
                                    shortDistY=0.05/self._geom.time2cm(),
                                    longDistY=1.0/self._geom.time2cm(),
                                    )

            thisPolyF = QtGui.QPolygonF(points)
            thisPoly = QtGui.QGraphicsPolygonItem(thisPolyF)
            thisPoly.setBrush(pg.mkColor((255,255,255,0))) # white

            thisPoly.setToolTip('Neutrino Interaction Vertex')

            self._drawnObjects[view.plane()].append(thisPoly)
            view._view.addItem(thisPoly)


        # TODO find a home for this block
        '''
        mb = view_manager.getMessageBar()
        message = str()
        tooltip = str()

        if mct.origin() == 1:
            pdg = pdg_to_name.get(mct.nu_pdg(), mct.nu_pdg())
            mode = mode_to_name.get(mct.int_mode(), mct.int_mode())
            message += f'PDG: {mct.nu_pdg()}, Neutrino Energy: {mct.nu_energy():.3} GeV, mode: {mode}'

            fs_pdgs = mct.finalstate_pdg()
            fs_enes = mct.finalstate_energy()
            tooltip += f'Vertex: x = {vertex[0]:.2f}, y = {vertex[1]:.2f}, z = {vertex[2]:.2f}\n\n'
            tooltip += 'Final State Particles'
            for p, e in zip(fs_pdgs, fs_enes):
                tooltip += f'\n  PDG: {p}, Energy: {e:.3} GeV'

        elif mct.origin() == 2:
            message += f'Cosmic Origin'
        elif mct.origin() == 3:
            message += f'Supernovae Event'
        elif mct.origin() == 4:
            # message += f'Single Particle Generation'
            if len(mct.finalstate_pdg()) == 1:
                message += f'Single particle. PDG: {mct.nu_pdg()}, Energy: {mct.nu_energy():.3} GeV.'
            else:
                message += f'Particle Gun Generation with {len(mct.finalstate_pdg())} particles'

                fs_pdgs = mct.finalstate_pdg()
                fs_enes = mct.finalstate_energy()
                tooltip += 'All Particles:'
                for p, e in zip(fs_pdgs, fs_enes):
                    tooltip += f'\n  PDG: {p}, Energy: {e:.3} GeV'



        mb.showMessage(message)
        mb.setToolTip(tooltip)
        '''

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


    def clearDrawnObjects(self, obj_list=None):
        """ Override base class since our object list is nested """
        for view_objs in self._drawnObjects:
            for obj in view_objs:
                obj.scene().removeItem(obj)
        self._drawnObjects = []
