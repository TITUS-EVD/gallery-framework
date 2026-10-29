from titus.display import wire_tick_positions
from titus.drawables.drawable import ReaderDrawable
from titus.drawables.items import Marker


class EndPoint2D(ReaderDrawable):
    '''
    recob::EndPoint2D: points on a wire plane (wire, drift time in ticks),
    placed like hits and drawn as fixed-size markers
    '''
    KIND = 'endpoints2d'
    COLOR = (0, 0, 0)
    RADIUS = 4  # pixels

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'endpoint2d'
        self._geom = geom
        self._module = tpc_module

    def drawObjects(self):
        for view in self.views():
            layout = self.view_layout(view)
            items = []
            for points in self._data:
                index, x, y = wire_tick_positions(layout, points.cryostat, points.tpc, points.plane, points.wire,
                                                  points.drift_time, self._geom.timeOffsetTicks(view.plane()))
                for i, px, py in zip(index, x, y):
                    marker = Marker(px, py, self.RADIUS, self.COLOR)
                    marker.setToolTip(f'End point {points.id[i]}: wire {points.wire[i]}, '
                                      f'time {points.drift_time[i]:.1f}, strength {points.strength[i]:.0f}')
                    items.append(marker)
            self.add_to_view(view, items)
