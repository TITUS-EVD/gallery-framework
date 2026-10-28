import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

from titus.display import ViewLayout, point_correction, position_tpcs, project_cone
from titus.drawables.drawable import ReaderDrawable
from titus.readers import Points

# recob::Shower::OpenAngle as the cone's full opening angle; the half-angle
# drawn is OPEN_ANGLE_TO_HALF_ANGLE times it
OPEN_ANGLE_TO_HALF_ANGLE = 0.5


class Shower(ReaderDrawable):
    '''
    recob::Shower cones (start, direction, length, opening angle) projected
    into the wire views: the outline of the projected 3D cone, drawn in the
    TPC of its start. Corrected (the TPC module's "Correct 3D objects"
    option), the shower's PFParticle T0 is undone and it is drawn in the TPC
    of its particle's spacepoint nearest its start. Corrected showers have a
    dashed outline, and their tooltip says what was changed.
    '''
    KIND = 'showers'
    COLORS = [
        (0, 147, 147, 100),  # dark teal
        (0, 0, 252, 100),  # bright blue
        (156, 0, 156, 100),  # purple
        (255, 0, 255, 100),  # pink
        (255, 0, 0, 100),  # red
        (175, 0, 0, 100),  # red/brown
        (252, 127, 0, 100),  # orange
        (102, 51, 0, 100),  # brown
        (127, 127, 127, 100),  # dark gray
        (210, 210, 210, 100),  # gray
        (100, 253, 0, 100),  # bright green
    ]

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'shower'
        self._geom = geom
        self._module = tpc_module
        self._corrections = []

    @staticmethod
    def starts(showers):
        return Points(id=showers.id, x=showers.start_x, y=showers.start_y, z=showers.start_z)

    def on_data(self):
        layout = ViewLayout(self._geom, plane=0)
        self._corrections = []
        for tag, showers in zip(self.producers(), self._data):
            starts = self.starts(showers)
            t0 = self.pfp_object_t0s('pfp_showers', str(tag), len(showers))
            tpcs = self.particle_tpcs('pfp_showers', str(tag), starts.xyz)
            self._corrections.append(point_correction(layout, starts, t0, tpcs, tpc_source="particle's"))

    def corrected(self):
        return getattr(self._module, 'correct_3d_objects', lambda: True)()

    def drawObjects(self):
        corrected = self.corrected()
        for view in self._module._wire_views.values():
            layout = ViewLayout(self._geom, plane=view.plane(), cryostat=view.cryostat())
            for showers, correction in zip(self._data, self._corrections):
                starts = self.starts(showers).xyz
                stored_tpcs = position_tpcs(self._geom.detector, starts)
                for i in range(len(showers)):
                    if not showers.length[i] > 0:
                        continue
                    if corrected:
                        shift, tpc, moved = correction.tick_shift[i], correction.tpcs[i], correction.moved[i]
                    else:
                        shift, tpc, moved = 0., stored_tpcs[i], False
                    direction = (showers.dir_x[i], showers.dir_y[i], showers.dir_z[i])
                    half_angle = OPEN_ANGLE_TO_HALF_ANGLE * showers.open_angle[i]
                    color = self.COLORS[i % len(self.COLORS)]
                    for polygon in project_cone(layout, starts[i], direction, showers.length[i], half_angle,
                                                shift, tpc):
                        item = QtWidgets.QGraphicsPolygonItem(
                            QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in zip(polygon.x, polygon.y)]))
                        item.setBrush(pg.mkBrush(color))
                        item.setPen(pg.mkPen(color[:3], width=1,
                                             style=QtCore.Qt.DashLine if moved else QtCore.Qt.SolidLine))
                        item.setToolTip(self.genToolTip(showers, i, correction if corrected else None))
                        view._view.addItem(item)
                        self._drawnObjects.append(item)

    def genToolTip(self, showers, i, correction):
        what = correction.describe(i) if correction is not None else 'drawn as stored'
        return (f'Shower {showers.id[i]}: length {showers.length[i]:.1f} cm, '
                f'opening angle {np.degrees(showers.open_angle[i]):.1f} deg, best plane {showers.best_plane[i]}\n'
                f'{what}')
