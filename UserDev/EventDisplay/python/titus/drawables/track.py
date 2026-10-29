import numpy as np
from PyQt5 import QtCore

from titus.display import track_correction, track_segments
from titus.display.projection import AS_STORED
from titus.drawables.drawable import ReaderDrawable
from titus.drawables.items import TrackCollection, TrackLine


class Track(ReaderDrawable):
    '''
    recob::Track trajectories projected into the wire views
    (titus.display.projection). Drawn as stored, or corrected to lie on
    their hits, as the TPC module's "Correct 3D objects" option says:
    Pandora's T0 shift is undone and each point is drawn in its hits' TPC,
    using the T0s and track-hit associations in the file. Corrected tracks
    are dashed, and their tooltip says what was changed.
    '''
    KIND = 'tracks'
    COLOR = (130, 0, 0)

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'track'
        self._geom = geom
        self._module = tpc_module
        # per producer: what the corrections need, and the corrections once computed
        self._inputs = []
        self._corrections = []

    def on_data(self):
        self._inputs = [self._correction_inputs(tag, tracks) for tag, tracks in zip(self.producers(), self._data)]
        self._corrections = [{} for _ in self._data]

    def _correction_inputs(self, tag, tracks):
        ''' T0 per track and the track-hit associations, from the products related to these tracks '''
        track_hits, hits = self.associated_hits('track_hits', str(tag))
        return {'t0': self.pfp_object_t0s('pfp_tracks', str(tag), len(tracks)),
                'track_hits': track_hits, 'hits': hits}

    def _correction(self, producer, row, layout):
        cache = self._corrections[producer]
        if row not in cache:
            inputs = self._inputs[producer]
            cache[row] = track_correction(layout, self._data[producer], row, inputs['t0'][row],
                                          inputs['track_hits'], inputs['hits'])
        return cache[row]

    def color(self, producer, row):
        return self.COLOR

    def corrected(self):
        return getattr(self._module, 'correct_3d_objects', lambda: True)()

    def drawObjects(self):
        corrected = self.corrected()
        for view in self.views():
            layout = self.view_layout(view)
            items = []
            for producer, tracks in enumerate(self._data):
                for row in range(len(tracks)):
                    correction = self._correction(producer, row, layout) if corrected else None
                    moved = correction is not None and correction.corrected
                    tooltip = None
                    for segment in track_segments(layout, tracks, row, correction):
                        if tooltip is None:
                            tooltip = self.genToolTip(tracks, row, correction)
                        items.append(TrackLine(np.stack([segment.x, segment.y], axis=1), self.color(producer, row),
                                               QtCore.Qt.DashLine if moved else QtCore.Qt.SolidLine,
                                               tooltip=tooltip, key=(producer, row)))
            # one item per view for all the lines: hover and repaints don't scale with their number
            if items:
                self.add_to_view(view, [TrackCollection(items)])

    def lines(self):
        ''' the TrackLines drawn, in all views '''
        return [line for collection in self._drawnObjects for line in collection.lines]

    def genToolTip(self, tracks, row, correction):
        points = tracks.points(row)
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum() if len(points) > 1 else 0.
        what = correction.describe() if correction is not None else AS_STORED
        return f'Track {tracks.id[row]}: length {length:.1f} cm, {len(points)} points\n{what}'
