import numpy as np

from titus.display import mc_t0, track_correction, track_segments
from titus.display.mctruth import MATCH_TOLERANCE_MEV, match_mctracks, particle_name
from titus.display.projection import AS_STORED
from titus.drawables.annotation import annotation_items
from titus.drawables.drawable import add_items, remove_items
from titus.drawables.track import Track
from titus.readers import Tracks

# straight paths (particles without a trajectory) are projected through points this far apart (cm)
STRAIGHT_STEP_CM = 1.


def mctracks_as_tracks(mctracks):
    ''' sim::MCTrack steps as Tracks trajectories (every step valid), for the track projection '''
    return Tracks(id=mctracks.track_id, point_offset=mctracks.step_offset, x=mctracks.x, y=mctracks.y,
                  z=mctracks.z, valid=np.ones(len(mctracks.x), dtype=bool))


def straight_path(mctracks, key, own_rows):
    '''
    (start, end, G4 start time, MCTrack row) of a generator particle drawn as
    a straight line, without a trajectory: its MCTrack's own start and end (a
    charged particle mcreco kept no steps of), else, for a particle without
    an MCTrack (neutral), the mother's start and end of a daughter whose
    mother it is (e.g. a Λ's decay proton). None if neither is known.
    '''
    def usable(start, end):
        return np.all(np.isfinite(start)) and np.all(np.isfinite(end)) and not np.allclose(start, end)

    for row in own_rows:
        if usable(mctracks.start(row), mctracks.end(row)):
            return mctracks.start(row), mctracks.end(row), float(mctracks.start_t[row]), row
    daughters = np.flatnonzero((mctracks.mother_pdg == key.pdg) & (mctracks.mother_process == 'primary')
                               & (mctracks.process != 'primary') & (mctracks.origin == key.origin)
                               & (np.abs(mctracks.mother_start_e - key.energy) < MATCH_TOLERANCE_MEV))
    for row in daughters:
        if usable(mctracks.mother_start(row), mctracks.mother_end(row)):
            return mctracks.mother_start(row), mctracks.mother_end(row), float(mctracks.mother_start_t[row]), row
    return None


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
        # no trajectory drawn (neutral, or no steps kept): a dashed straight line
        for key in keys:
            if key in drawn:
                continue
            ids = self._annotate_straight(key, rows)
            if ids:
                drawn.add(key)
                found[key] = ids
        return {key: ids for key, ids in found.items() if key in drawn}

    def _annotate_straight(self, key, rows):
        '''
        draw a particle without a trajectory as a dashed straight line, in
        each view (rows: {(producer, row): key} of the matched MCTracks);
        [MCTrack ID] it was found from, or []
        '''
        for producer, stored in enumerate(self._stored_data):
            own = [r for (p, r), k in rows.items() if p == producer and k == key]
            path = straight_path(stored, key, own)
            if path is None:
                continue
            start, end, g4_time, row = path
            n = int(np.ceil(np.linalg.norm(end - start) / STRAIGHT_STEP_CM)) + 1
            xyz = start + np.linspace(0, 1, min(max(n, 2), 5000))[:, None] * (end - start)
            if self.space_charge_applied(producer):
                xyz = self._sce_map.displace(xyz)
            line = Tracks(id=[int(stored.track_id[row])], point_offset=[0, len(xyz)], x=xyz[:, 0], y=xyz[:, 1],
                          z=xyz[:, 2], valid=np.ones(len(xyz), dtype=bool))
            name, drawn = particle_name(key.pdg), False
            for view in self.views():
                layout = self.view_layout(view)
                correction = track_correction(layout, line, 0, t0=mc_t0(self._geom, g4_time)) if self.corrected() else None
                segments = [np.stack([s.x, s.y], axis=1) for s in track_segments(layout, line, 0, correction)]
                items = annotation_items(segments, name, dashed=True)
                if items:
                    add_items(view._view, items)
                    self._annotations.extend(items)
                    drawn = True
            if drawn:
                return [int(stored.track_id[row])]
        return []

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
