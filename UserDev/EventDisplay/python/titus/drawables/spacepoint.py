import numpy as np
import pyqtgraph as pg

from titus.display import ViewLayout, object_hit_tpcs, point_correction, project_objects
from titus.drawables.drawable import ReaderDrawable


class SpacePoint(ReaderDrawable):
    '''
    recob::SpacePoint positions projected into the wire views, as one
    scatter plot per view with fixed-size symbols (there are thousands).
    Corrected (the TPC module's "Correct 3D objects" option), a spacepoint's
    PFParticle T0 is undone and it is drawn in the TPC of its hits. Corrected
    spacepoints are hollow.
    '''
    KIND = 'spacepoints'
    COLOR = (255, 0, 255)
    SIZE = 4  # pixels

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'spacepoint'
        self._geom = geom
        self._module = tpc_module
        self._corrections = []

    def on_data(self):
        layout = ViewLayout(self._geom, plane=0)
        self._corrections = []
        for tag, points in zip(self.producers(), self._data):
            t0 = self.pfp_object_t0s('pfp_spacepoints', str(tag), len(points))
            spacepoint_hits, hits = self.associated_hits('spacepoint_hits', str(tag))
            hit_tpcs = object_hit_tpcs(len(points), spacepoint_hits, hits) if hits is not None else None
            self._corrections.append(point_correction(layout, points, t0, hit_tpcs))

    def corrected(self):
        return getattr(self._module, 'correct_3d_objects', lambda: True)()

    def drawObjects(self):
        corrected = self.corrected()
        for view in self._module._wire_views.values():
            layout = ViewLayout(self._geom, plane=view.plane(), cryostat=view.cryostat())
            for points, correction in zip(self._data, self._corrections):
                use = correction if corrected else None
                projected = project_objects(layout, points, use).values()
                if not projected:
                    continue
                index = np.concatenate([p[0] for p in projected])
                x = np.concatenate([p[1] for p in projected])
                y = np.concatenate([p[2] for p in projected])
                moved = correction.moved[index] if use is not None else np.zeros(len(index), dtype=bool)
                for selection, hollow in ((~moved, False), (moved, True)):
                    if not selection.any():
                        continue
                    scatter = pg.ScatterPlotItem(
                        x[selection], y[selection], size=self.SIZE, pxMode=True,
                        pen=pg.mkPen(self.COLOR) if hollow else pg.mkPen(None),
                        brush=pg.mkBrush(None) if hollow else pg.mkBrush(self.COLOR))
                    view._view.addItem(scatter)
                    self._drawnObjects.append(scatter)
