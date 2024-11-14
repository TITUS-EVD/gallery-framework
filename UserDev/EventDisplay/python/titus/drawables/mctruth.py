from titus.drawables import Drawable
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from ROOT import evd, larutil
import pyqtgraph as pg

# Maximum number of final state particles to be displayed
MAX_FS_PARS = 6

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

inttype_to_name = {
    0: 'CC',
    1: 'NC'
}

class MCTruth(Drawable):

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'mctruth'
        self._process = evd.DrawMCTruth(geom.getGeometryCore(), geom.getDetectorProperties(), geom.getDetectorClocks())
        self._geom = geom
        self._module = tpc_module
        self._module._show_vertex.stateChanged.connect(self.toggle_cross)
        self._mcts = None
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

        self._mcts = self._process.getData()

        if len(self._mcts) == 0:
            return

        # Start by drawing the fist one
        self.draw_one_mctruth(self._mcts[0])

        self._module._mctruth_dropdown.clear()
        for i in range(len(self._mcts)):
            self._module._mctruth_dropdown.addItem(f'Neutrino {i+1}')
        self._module._mctruth_dropdown.currentIndexChanged.connect(self.on_selection_changed)


    def on_selection_changed(self, index):

        # Clear the current objects
        self.clearDrawnObjects(hide=False)

        # Draw the requestes mctruth
        self.draw_one_mctruth(self._mcts[index])


    def draw_one_mctruth(self, mct):
        '''
        Draws a single MCTruth object
        '''

        vertex = mct.vertex()

        for _, view in self._module._wire_views.items():
            self._drawnObjects.append([])

            geo_helper = larutil.SimpleGeometryHelper(self._geom.getGeometryCore(),
                                                      self._geom.getDetectorProperties(),
                                                      self._geom.getDetectorClocks())

            tpc = 0 if vertex[0] < 0 else 1
            plane = view.plane()

            if not self._geom.projectionsMatch():
                # swap plane 0 and 1 for TPC 1
                if tpc == 1:
                    if plane != 2:
                        plane =  abs(plane - 1)

            vertexPoint = geo_helper.Point_3Dto2D(vertex, plane)

            # Convert to wires and ticks
            vertexPoint_2d_w = vertexPoint.w/self._geom.wire2cm()
            vertexPoint_2d_t = vertexPoint.t/self._geom.time2cm()

            if tpc == 1:
                # flip
                vertexPoint_2d_t = self._geom.tRange() - vertexPoint_2d_t
                # move up
                vertexPoint_2d_t += self._geom.tRange()
                # add cathode gap
                vertexPoint_2d_t += self._geom.cathodeGap()



            points = self.makeCross(startX=vertexPoint_2d_w,
                                    startY=vertexPoint_2d_t,
                                    shortDistX=1.0/self._geom.wire2cm(),
                                    longDistX=4.0/self._geom.wire2cm(),
                                    shortDistY=1.0/self._geom.time2cm(),
                                    longDistY=4.0/self._geom.time2cm(),
                                    )

            thisPolyF = QtGui.QPolygonF(points)
            thisPoly = QtWidgets.QGraphicsPolygonItem(thisPolyF)
            # thisPoly.setBrush(pg.mkColor((255,255,255,0))) # white
            thisPoly.setBrush(QtGui.QBrush(QtGui.QColor("salmon"))) # white

            thisPoly.setToolTip('Neutrino Interaction Vertex')

            self._drawnObjects[view.plane()].append(thisPoly)
            view._view.addItem(thisPoly)


        message = str()
        tooltip = str()

        if mct.origin() == 1:
            pdg = pdg_to_name.get(mct.nu_pdg(), 'NA')
            mode = mode_to_name.get(mct.int_mode(), 'NA')
            inttype = inttype_to_name.get(mct.ccnc(), 'NA')
            message += f'PDG: {mct.nu_pdg()}, {inttype}, E: {mct.nu_energy():.3} GeV, mode: {mode}'

            fs_pdgs = mct.finalstate_pdg()
            fs_enes = mct.finalstate_energy()

            fs_pars_dropped = False
            if len(fs_pdgs) > MAX_FS_PARS:
                fs_pars_dropped = True
                fs_pdgs = fs_pdgs[:MAX_FS_PARS]
                fs_enes = fs_enes[:MAX_FS_PARS]

            tooltip += f'Vertex: x = {vertex[0]:.2f}, y = {vertex[1]:.2f}, z = {vertex[2]:.2f}\n'
            tooltip += 'Final State Particles:'
            for p, e in zip(fs_pdgs, fs_enes):
                tooltip += f'\n  PDG: {p}, Energy: {e:.3} GeV'
            if fs_pars_dropped:
                tooltip += f'\n  ... [some particles not shown]'



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


        self._module._mctruth_text1.setText(message)
        self._module._mctruth_text1.setToolTip(tooltip)
        self._module._mctruth_text2.setText(tooltip)
        self._module._mctruth_dock.show()

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

    def toggle_cross(self):
        if self._module._show_vertex.isChecked():
            for view_objs in self._drawnObjects:
                for obj in view_objs:
                    obj.show()
        else:
            for view_objs in self._drawnObjects:
                for obj in view_objs:
                    obj.hide()


    def clearDrawnObjects(self, obj_list=None, hide=True):
        """ Override base class since our object list is nested """
        for view_objs in self._drawnObjects:
            for obj in view_objs:
                obj.scene().removeItem(obj)
        self._drawnObjects = []
        if hide:
            self._module._mctruth_dock.hide()
