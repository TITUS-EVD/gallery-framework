import numpy as np

from titus.display import ViewLayout, point_correction, project_objects
from titus.drawables.drawable import ReaderDrawable
from titus.drawables.items import Marker


class Vertex(ReaderDrawable):
    '''
    recob::Vertex positions projected into the wire views, as fixed-size
    markers. Corrected (the TPC module's "Correct 3D objects" option), a
    vertex's PFParticle T0 is undone and it is drawn in the TPC whose readout
    window contains it; vertices have no hits to place them by. Corrected
    vertices are hollow, and their tooltip says what was changed.
    '''
    KIND = 'vertices'
    COLOR = (0, 255, 255)
    RADIUS = 5  # pixels

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'vertex'
        self._geom = geom
        self._module = tpc_module
        self._t0s = []
        self._corrections = []

    def on_data(self):
        self._t0s = [self.pfp_object_t0s('pfp_vertices', str(tag), len(vertices))
                     for tag, vertices in zip(self.producers(), self._data)]
        layout = ViewLayout(self._geom, plane=0)
        self._corrections = [point_correction(layout, vertices, t0)
                             for vertices, t0 in zip(self._data, self._t0s)]

    def corrected(self):
        return getattr(self._module, 'correct_3d_objects', lambda: True)()

    def drawObjects(self):
        corrected = self.corrected()
        for view in self._module._wire_views.values():
            layout = ViewLayout(self._geom, plane=view.plane(), cryostat=view.cryostat())
            for vertices, correction in zip(self._data, self._corrections):
                use = correction if corrected else None
                for _, (index, x, y) in project_objects(layout, vertices, use).items():
                    for i, vx, vy in zip(index, x, y):
                        moved = use is not None and correction.moved[i]
                        marker = Marker(vx, vy, self.RADIUS, self.COLOR, filled=not moved)
                        marker.setToolTip(self.genToolTip(vertices, i, use))
                        view._view.addItem(marker)
                        self._drawnObjects.append(marker)

    def genToolTip(self, vertices, i, correction):
        what = correction.describe(i) if correction is not None else 'drawn as stored'
        x, y, z = vertices.xyz[i]
        return f'Vertex {vertices.id[i]}: ({x:.1f}, {y:.1f}, {z:.1f}) cm\n{what}'
