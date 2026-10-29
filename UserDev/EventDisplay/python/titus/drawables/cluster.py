import numpy as np

from titus.display import hit_positions
from titus.drawables.drawable import ReaderDrawable
from titus.drawables.hit import HitCollection
from titus.readers import Hits


class Cluster(ReaderDrawable):
    '''
    recob::Cluster as its hits (the cluster_hits association), drawn like
    hits (one wire wide, peak time +- RMS) in a color per cluster. Hovering a
    hit outlines its whole cluster. One item per view, as for hits.
    '''
    KIND = 'clusters'
    COLORS = [
        (0, 147, 147),  # dark teal
        (0, 0, 252),    # bright blue
        (156, 0, 156),  # purple
        (255, 0, 255),  # pink
        (255, 0, 0),    # red
        (175, 0, 0),    # red/brown
        (252, 127, 0),  # orange
        (102, 51, 0),   # brown
        (127, 127, 127),  # dark gray
        (210, 210, 210),  # gray
        (100, 253, 0),  # bright green
    ]
    ALPHA = 200

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'cluster'
        self._geom = geom
        self._module = tpc_module
        # per producer: (cluster row of each hit, the hits), or None without associations
        self._cluster_hits = []

    def on_data(self):
        self._cluster_hits = []
        for tag, clusters in zip(self.producers(), self._data):
            assns, hits = self.associated_hits('cluster_hits', str(tag))
            if assns is None:
                print(f'Warning: no cluster-hit associations for {tag}: clusters not drawn')
                self._cluster_hits.append(None)
                continue
            rows = assns.right
            self._cluster_hits.append((assns.left, Hits(**{k: v[rows] for k, v in hits.to_arrays().items()})))

    def drawObjects(self):
        for view in self.views():
            layout = self.view_layout(view)
            for clusters, cluster_hits in zip(self._data, self._cluster_hits):
                if cluster_hits is None or len(cluster_hits[1]) == 0:
                    continue
                owner, hits = cluster_hits
                positions = hit_positions(layout, hits, time_offset_ticks=self._geom.timeOffsetTicks(view.plane()))
                if len(positions) == 0:
                    continue
                cluster = owner[positions.index]
                colors = np.array([self.COLORS[c % len(self.COLORS)] + (self.ALPHA,) for c in cluster])
                item = HitCollection(positions.x - 0.5, positions.y - positions.half_height, np.ones(len(positions)),
                                     2 * positions.half_height, colors=colors, group=cluster,
                                     tooltip=lambda i, clusters=clusters, cluster=cluster:
                                     self.genToolTip(clusters, cluster[i]))
                view._view.addItem(item)
                self._drawnObjects.append(item)

    def genToolTip(self, clusters, row):
        return (f'Cluster {clusters.id[row]}: {clusters.n_hits[row]} hits, '
                f'wires {clusters.start_wire[row]:.0f}-{clusters.end_wire[row]:.0f}, '
                f'integral {clusters.integral[row]:.0f}')
