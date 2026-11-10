import numpy as np

from titus.display import mc_t0, track_correction, track_segments
from titus.display.mctruth import MATCH_TOLERANCE_MEV, g4_pdgs, match_mctracks, particle_name, primary_daughters
from titus.display.projection import AS_STORED
from titus.drawables.annotation import FONT_POINT_SIZE, annotation_items
from titus.drawables.drawable import ReaderDrawable, add_items, remove_items
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
    mother it is (e.g. a Λ's decay proton; a K⁰'s, the K⁰S or K⁰L Geant4
    made it, g4_pdgs). None if neither is known.
    '''
    def usable(start, end):
        return np.all(np.isfinite(start)) and np.all(np.isfinite(end)) and not np.allclose(start, end)

    for row in own_rows:
        if usable(mctracks.start(row), mctracks.end(row)):
            return mctracks.start(row), mctracks.end(row), float(mctracks.start_t[row]), row
    daughters = np.flatnonzero(np.isin(mctracks.mother_pdg, g4_pdgs(key.pdg)) & primary_daughters(mctracks)
                               & (mctracks.origin == key.origin)
                               & (np.abs(mctracks.mother_start_e - key.energy) < MATCH_TOLERANCE_MEV))
    for row in daughters:
        if usable(mctracks.mother_start(row), mctracks.mother_end(row)):
            return mctracks.mother_start(row), mctracks.mother_end(row), float(mctracks.mother_start_t[row]), row
    return None


class MCTrack(Track):
    '''
    sim::MCTrack: simulated particle paths, projected like recob::Track.
    The particles selected in the MC truth dock (TpcModule.select_mc_particles)
    are highlighted; those annotated there are drawn by MCTrackAnnotations,
    with or without these.
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
        self._stored_data = []

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
        # the MC truth dock's note ("turn on MCTrack"...) for the tracks now drawn
        refresh = getattr(self._module, 'refresh_mctruth_notes', None)
        if callable(refresh):
            refresh()

    def _rows_of(self, keys):
        ''' ({(producer, row): key} of the MCTracks of these particles, {key: [MCTrack IDs]}) '''
        rows, found = {}, {}
        for producer, mctracks in enumerate(self._mctracks):
            for key, matched in match_mctracks(mctracks, keys).items():
                rows.update({(producer, r): key for r in matched})
                found.setdefault(key, []).extend(int(mctracks.track_id[r]) for r in matched)
        return rows, found

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


class MCTrackAnnotations(MCTrack):
    '''
    The MC truth dock's annotations: the annotated particles' paths in white
    with their names (titus.drawables.annotation), and what the particles
    with their Secondaries box checked made. It reads its own MCTracks for
    the TPC module (not a product box: annotating needs no MCTracks drawn),
    only once something is annotated, and projects the annotated ones only.
    The module says what and how: annotated_mc_particles,
    annotated_mc_secondaries, annotated_mc_paths (None: all drawn with
    their paths), annotation_font_size.
    '''

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, geom, tpc_module, *args, **kwargs)
        # graphics items of the annotated particles, in the views
        self._annotations = []
        # labels moved by the user: {(ParticleKey, (plane, cryostat), mode, MCTrack ID): (x, y)}
        self._label_positions = {}
        # this event's MCTracks not read yet
        self._stale = True

    def set_producer(self, producer):
        before = self.producers()
        super().set_producer(producer)
        if self.producers() != before:
            self._stale = True

    def analyze(self):
        # another event, or read again (e.g. space charge): read when something is annotated
        self.clearDrawnObjects()
        self._stale = True

    def _read_if_stale(self):
        if self._stale:
            self._stale = False
            ReaderDrawable.analyze(self)

    def on_data(self):
        # another event: labels where they go
        self._label_positions = {}
        super().on_data()

    def drawObjects(self):
        self.annotate_particles(self._module_keys('annotated_mc_particles'),
                                self._module_keys('annotated_mc_secondaries'),
                                self._module_keys('annotated_mc_paths', default=None))

    def _module_keys(self, name, default=()):
        getter = getattr(self._module, name, None)
        keys = getter() if callable(getter) else None
        return keys if isinstance(keys, list) else (None if default is None else [])

    def _font_size(self):
        ''' of the labels, as the module says '''
        size = getattr(self._module, 'annotation_font_size', None)
        size = size() if callable(size) else None
        return size if isinstance(size, int) and size > 0 else FONT_POINT_SIZE

    def _rotated(self):
        ''' whether the labels are written along their paths (else level), as the module says '''
        rotated = getattr(self._module, 'annotation_labels_rotated', None)
        return (rotated() if callable(rotated) else True) is not False

    def clearDrawnObjects(self, obj_list=None):
        super().clearDrawnObjects(obj_list)
        self._clear_annotations()

    def drawn_items(self):
        ''' the graphics items of the annotations drawn, in the views '''
        return list(self._annotations)

    def _clear_annotations(self):
        remove_items(self._annotations)
        self._annotations = []

    def annotate_particles(self, keys, secondaries=(), paths=None):
        '''
        annotate these generator particles (ParticleKeys) in each view, in
        white with their names: the MCTrack's path as MCTrack draws it
        (corrected, with space charge), else (neutral, or no steps kept) a
        dashed straight line from its start to its end; and the descendants
        (G4 parentage) of the particles in secondaries, annotated themselves
        or not. paths: the particles whose annotations (theirs and their
        secondaries') are drawn with their paths, the others names only;
        None: all. Labels can be moved: their positions are kept while the
        particle stays annotated. {key: [MCTrack IDs]} of the particles of
        keys drawn in a view
        '''
        self._clear_annotations()
        keys, secondaries = list(keys), list(secondaries)
        wanted = list(dict.fromkeys(keys + secondaries))
        self._label_positions = {k: v for k, v in self._label_positions.items() if k[0] in wanted}
        if not wanted:
            return {}
        self._read_if_stale()
        rows, _ = self._rows_of(wanted)
        font_size, rotate = self._font_size(), self._rotated()
        # {(view, producer, row): view points of its segments}, projected once
        projected = {}
        drawn = {}
        for key in wanted:
            style = (font_size, paths is None or key in paths, rotate)
            for producer, mctracks in enumerate(self._stored_data):
                own = [r for (p, r), k in rows.items() if p == producer and k == key]
                primary, primary_id = self._primary_target(mctracks, key, own)
                if key in keys and primary is not None and \
                        self._annotate(key, producer, mctracks, *primary, projected, style):
                    drawn.setdefault(key, []).append(int(mctracks.track_id[primary[1]]))
                if key in secondaries and primary_id is not None:
                    for r in self.descendants(mctracks, primary_id):
                        self._annotate(key, producer, mctracks, 'own', int(r), mctracks.pdg[r], projected, style)
        return drawn

    def _segments(self, view, producer, row, paths):
        ''' view points of the segments of an MCTrack's path in a view, as MCTrack draws them (kept in paths) '''
        if (id(view), producer, row) not in paths:
            layout = self.view_layout(view)
            correction = self._correction(producer, row, layout) if self.corrected() else None
            paths[(id(view), producer, row)] = [
                np.stack([s.x, s.y], axis=1) for s in track_segments(layout, self._data[producer], row, correction)]
        return paths[(id(view), producer, row)]

    @staticmethod
    def _primary_target(mctracks, key, own):
        '''
        (what to draw, its G4 track ID) of a generator particle: its own
        MCTrack ('own', row, PDG code), else the mother's start and end of a
        daughter of it ('mother', row, PDG code), e.g. a Λ's; (None, None)
        '''
        if own:
            return ('own', own[0], key.pdg), int(mctracks.track_id[own[0]])
        path = straight_path(mctracks, key, [])
        if path is None:
            return None, None
        row = path[3]
        primary_id = mctracks.ancestor_track_id[row] if mctracks.ancestor_track_id[row] >= 0 \
            else mctracks.mother_track_id[row]
        return ('mother', row, key.pdg), (int(primary_id) if primary_id >= 0 else None)

    @staticmethod
    def descendants(mctracks, primary_id):
        ''' rows of the MCTracks descending from a primary (their ancestor's G4 track ID), the primary excluded '''
        return np.flatnonzero((mctracks.ancestor_track_id == primary_id) & (mctracks.track_id != primary_id))

    def _annotate(self, key, producer, mctracks, mode, row, pdg, paths, style):
        '''
        annotate one MCTrack (mode 'own') or the mother of one ('mother') in
        each view: its trajectory as drawn, else a dashed straight line from
        start to end. style: (font size, paths drawn, labels rotated). True if drawn in a view.
        '''
        if mode == 'own':
            start, end, g4_time = mctracks.start(row), mctracks.end(row), float(mctracks.start_t[row])
        else:
            start, end, g4_time = mctracks.mother_start(row), mctracks.mother_end(row), float(mctracks.mother_start_t[row])
        font_size, show_line, rotate = style
        straight = None
        drawn = False
        for view in self.views():
            segments = self._segments(view, producer, row, paths) if mode == 'own' else None
            dashed = not segments
            if dashed:
                if straight is None:
                    straight = self._straight_line(producer, start, end, int(mctracks.track_id[row]))
                if straight is False:
                    continue
                segments = self._straight_segments(view, straight, g4_time)
            label_key = (key, (view.plane(), view.cryostat()), mode, int(mctracks.track_id[row]))
            items = annotation_items(segments, particle_name(pdg), dashed=dashed,
                                     label_pos=self._label_positions.get(label_key),
                                     on_moved=lambda pos, k=label_key: self._label_positions.__setitem__(k, pos),
                                     font_size=font_size, show_line=show_line, rotate=rotate)
            if items:
                add_items(view._view, items)
                self._annotations.extend(items)
                drawn = True
        return drawn

    def _straight_line(self, producer, start, end, track_id):
        ''' a straight line from start to end as a one-track Tracks (moved by space charge if the rest is), or False '''
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))) or np.allclose(start, end):
            return False
        n = int(np.ceil(np.linalg.norm(end - start) / STRAIGHT_STEP_CM)) + 1
        xyz = start + np.linspace(0, 1, min(max(n, 2), 5000))[:, None] * (end - start)
        if self.space_charge_applied(producer):
            xyz = self._sce_map.displace(xyz)
        return Tracks(id=[track_id], point_offset=[0, len(xyz)], x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                      valid=np.ones(len(xyz), dtype=bool))

    def _straight_segments(self, view, line, g4_time):
        layout = self.view_layout(view)
        correction = track_correction(layout, line, 0, t0=mc_t0(self._geom, g4_time)) if self.corrected() else None
        return [np.stack([s.x, s.y], axis=1) for s in track_segments(layout, line, 0, correction)]
