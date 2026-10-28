import numpy as np

from titus.display import ViewLayout, object_hit_tpcs, particle_tpcs, point_correction, project_objects
from titus.drawables.drawable import ReaderDrawable
from titus.drawables.items import Marker


class Vertex(ReaderDrawable):
    '''
    recob::Vertex positions projected into the wire views, as fixed-size
    markers. Corrected (the TPC module's "Correct 3D objects" option), a
    vertex's PFParticle T0 is undone and, as vertices have no hits, it is
    drawn in the TPC of its particle's nearest spacepoint's hit (else the TPC
    whose readout window contains it). Corrected vertices are hollow, and
    their tooltip says what was changed.
    '''
    KIND = 'vertices'
    COLOR = (0, 255, 255)
    RADIUS = 5  # pixels

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'vertex'
        self._geom = geom
        self._module = tpc_module
        self._corrections = []

    def on_data(self):
        layout = ViewLayout(self._geom, plane=0)
        self._corrections = []
        for tag, vertices in zip(self.producers(), self._data):
            t0 = self.pfp_object_t0s('pfp_vertices', str(tag), len(vertices))
            self._corrections.append(point_correction(layout, vertices, t0, self._particle_tpcs(str(tag), vertices),
                                                      tpc_source="particle's"))

    def _particle_tpcs(self, tag, vertices):
        ''' TPC per vertex from its PFParticle's spacepoints and their hits, or None '''
        pfp_vertices = self.read_optional('pfp_vertices', tag)
        if pfp_vertices is None or len(pfp_vertices) == 0:
            return None
        pfp_tag = pfp_vertices.product_tags[pfp_vertices.left_product[0]]
        pfp_spacepoints = self.read_optional('pfp_spacepoints', pfp_tag)
        if pfp_spacepoints is None or len(pfp_spacepoints) == 0:
            return None
        spacepoint_tag = pfp_spacepoints.product_tags[pfp_spacepoints.right_product[0]]
        spacepoints = self.read_optional('spacepoints', spacepoint_tag)
        spacepoint_hits, hits = self.associated_hits('spacepoint_hits', spacepoint_tag)
        if spacepoints is None or hits is None:
            return None
        return particle_tpcs(vertices.xyz, pfp_vertices, pfp_spacepoints, spacepoints,
                             object_hit_tpcs(len(spacepoints), spacepoint_hits, hits))

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
