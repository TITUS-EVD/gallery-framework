import numpy as np

from titus.display import mc_t0
from titus.display.mctruth import match_mctracks, particle_name
from titus.display.projection import AS_STORED
from titus.drawables.annotation import annotation_items
from titus.drawables.drawable import add_items, remove_items
from titus.drawables.track import Track
from titus.readers import Tracks


def mctracks_as_tracks(mctracks):
    ''' sim::MCTrack steps as Tracks trajectories (every step valid), for the track projection '''
    return Tracks(id=mctracks.track_id, point_offset=mctracks.step_offset, x=mctracks.x, y=mctracks.y,
                  z=mctracks.z, valid=np.ones(len(mctracks.x), dtype=bool))


class MCTrack(Track):
    '''
    sim::MCTrack: simulated particle paths, projected like recob::Track.
    The particles selected in the MC truth dock (TpcModule.select_mc_particles)
    are highlighted.
    Corrected (the TPC module's "Correct 3D objects" option), a particle's
    time relative to the trigger is its T0, so the path lands on the hits its
    charge made (out-of-time cosmics move), and each step is drawn in the TPC
    whose readout window contains it. Particles from a neutrino are gray,
    others black.
    '''
    KIND = 'mctracks'
    SIMULATION = True
    COLOR = (0, 0, 0)
    NEUTRINO_COLOR = (128, 128, 128)
    NEUTRINO_ORIGIN = 1  # simb::kBeamNeutrino

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, geom, tpc_module, *args, **kwargs)
        self._product_name = 'mctrack'
        self._mctracks = []
        # graphics items of the annotated particles, in the views
        self._annotations = []

    def on_data(self):
        self._mctracks = self._data
        self._data = [mctracks_as_tracks(m) for m in self._mctracks]
        self._inputs = [{'t0': mc_t0(self._geom, m.start_t), 'track_hits': None, 'hits': None}
                        for m in self._mctracks]
        self._corrections = [{} for _ in self._data]

    def drawObjects(self):
        super().drawObjects()
        selected = getattr(self._module, 'selected_mc_particles', None)
        keys = selected() if selected is not None else []
        self.highlight_particles(keys if isinstance(keys, list) else [])
        annotated = getattr(self._module, 'annotated_mc_particles', None)
        keys = annotated() if annotated is not None else []
        self.annotate_particles(keys if isinstance(keys, list) else [])
        # the MC truth dock's notes ("turn on MCTrack"...) for the tracks now drawn
        refresh = getattr(self._module, 'refresh_mctruth_notes', None)
        if callable(refresh):
            refresh()

    def clearDrawnObjects(self, obj_list=None):
        super().clearDrawnObjects(obj_list)
        self._clear_annotations()

    def _clear_annotations(self):
        remove_items(self._annotations)
        self._annotations = []

    def _rows_of(self, keys):
        ''' ({(producer, row): key} of the MCTracks of these particles, {key: [MCTrack IDs]}) '''
        rows, found = {}, {}
        for producer, mctracks in enumerate(self._mctracks):
            for key, matched in match_mctracks(mctracks, keys).items():
                rows.update({(producer, r): key for r in matched})
                found.setdefault(key, []).extend(int(mctracks.track_id[r]) for r in matched)
        return rows, found

    def annotate_particles(self, keys):
        '''
        annotate the MCTracks of these generator particles (ParticleKeys) in
        each view: their path, as drawn (corrected, with space charge), in
        white with the particle's name above it. {key: [MCTrack IDs]} of
        those drawn in a view (MCTracks without steps there aren't)
        '''
        self._clear_annotations()
        rows, found = self._rows_of(keys)
        drawn = set()
        for view, collection in self._view_collections:
            paths = {}
            for line in collection.lines:
                if line.track_key in rows:
                    paths.setdefault(line.track_key, []).append(line.points)
            items = []
            for track_key, segments in paths.items():
                annotation = annotation_items(segments, particle_name(rows[track_key].pdg))
                if annotation:
                    drawn.add(rows[track_key])
                    items += annotation
            if items:
                add_items(view._view, items)
                self._annotations.extend(items)
        return {key: ids for key, ids in found.items() if key in drawn}

    def highlight_particles(self, keys):
        '''
        highlight the MCTracks of these generator particles (ParticleKeys,
        titus.display.mctruth.match_mctracks); {key: [MCTrack IDs]} found
        '''
        selected, found = set(), {}
        for producer, mctracks in enumerate(self._mctracks):
            for key, rows in match_mctracks(mctracks, keys).items():
                selected.update((producer, r) for r in rows)
                found.setdefault(key, []).extend(int(mctracks.track_id[r]) for r in rows)
        for line in self.lines():
            line.set_selected(line.track_key in selected)
        return found

    def color(self, producer, row):
        origin = self._mctracks[producer].origin[row]
        return self.NEUTRINO_COLOR if origin == self.NEUTRINO_ORIGIN else self.COLOR

    def genToolTip(self, tracks, row, correction):
        mctracks = next(m for m, t in zip(self._mctracks, self._data) if t is tracks)
        what = correction.describe() if correction is not None else AS_STORED
        return (f'MCTrack {mctracks.track_id[row]}: PDG {mctracks.pdg[row]}, '
                f'{mctracks.start_e[row] / 1e3:.3f} GeV, {mctracks.process[row]}\n'
                f'time {mc_t0(self._geom, mctracks.start_t[row]) / 1e3:.2f} us from the trigger\n{what}'
                + self.space_charge_note(tracks))
