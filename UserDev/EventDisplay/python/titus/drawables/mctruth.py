import numpy as np

from titus.display import ViewLayout, mc_t0, point_correction, position_tpcs, project_objects
from titus.display.mctruth import neutrino_path, particle_key, particle_name, particle_rows, truth_label, \
    truth_summary
from titus.display.projection import AS_STORED, track_correction, track_segments
from titus.drawables.annotation import FONT_POINT_SIZE, annotation_items
from titus.drawables.drawable import ReaderDrawable, add_items, remove_items
from titus.drawables.items import CrossMarker
from titus.readers import Points, Tracks

# the neutrino's line is projected through points this far apart (cm)
NEUTRINO_STEP_CM = 1.


class MCTruth(ReaderDrawable):
    '''
    simb::MCTruth: the MC truth dock (titus.gui.mctruth_panel, in the TPC
    module) describes one interaction at a time, chosen in its dropdown
    (neutrinos first), and lists its particles. Every neutrino vertex is
    drawn as a cross in the wire views, the selected interaction's
    highlighted. Selecting particles in the dock highlights their MCTracks
    (TpcModule.select_mc_particles). Corrected (the TPC module's "Correct 3D
    objects" option), each interaction's time is undone like a T0, so its
    vertex lands where its charge was read out. With the dock's Neutrino
    path, the shown interaction's incoming neutrino is annotated: a dashed
    line from outside the active volume to the vertex, with an arrowhead
    there and its name (titus.drawables.annotation), corrected and moved by
    space charge as the vertex is.
    '''
    KIND = 'mctruths'
    SIMULATION = True

    def __init__(self, gallery_interface, geom, tpc_module, *args, **kwargs):
        super().__init__(gallery_interface, *args, **kwargs)
        self._product_name = 'mctruth'
        self._geom = geom
        self._module = tpc_module
        # (producer, row) of each dropdown entry, and the one shown
        self._choices = []
        self._selected = 0
        # (dropdown entry, cross) of the drawn vertices
        self._crosses = []
        # the interaction whose particles the dock lists, (truths, row)
        self._listed = None
        # the dock's notes on the selected and the annotated particles' MCTracks
        self._notes = {'selected': '', 'annotated': '', 'neutrino': ''}
        # the neutrino's annotation (graphics items, in the views), and where its labels were moved:
        # {(producer, row, (plane, cryostat)): (x, y)}
        self._nu_items = []
        self._nu_label_positions = {}
        panel = self.panel()
        if panel is not None:
            panel.dropdown.currentIndexChanged.connect(self.on_selection_changed)
            panel.show_vertex.stateChanged.connect(self.toggle_cross)
            panel.all_particles.toggled.connect(lambda _: self.show_particles())
            panel.particlesSelected.connect(self.select_particles)
            panel.particlesAnnotated.connect(self.on_particles_annotated)
            panel.show_neutrino.toggled.connect(lambda _: self.draw_neutrino())

    def panel(self):
        ''' the TPC module's MC truth panel (None in tests without one) '''
        return getattr(self._module, '_mctruth_panel', None)

    def on_data(self):
        self._nu_label_positions = {}
        self._choices = sorted(((p, row) for p, truths in enumerate(self._data) for row in range(len(truths))),
                               key=lambda c: not self._data[c[0]].has_neutrino[c[1]])
        self._selected = 0
        panel = self.panel()
        if panel is None:
            return
        panel.dropdown.blockSignals(True)
        panel.dropdown.clear()
        for p, row in self._choices:
            panel.dropdown.addItem(truth_label(self._data[p], row, str(self.producers()[p]).split(':')[0]))
        panel.dropdown.blockSignals(False)

    def set_producer(self, producer):
        super().set_producer(producer)
        if producer is None:
            self.hide_dock()

    def hide_dock(self):
        panel = self.panel()
        if panel is not None:
            panel.clear()
            self._module.show_mctruth_dock(False)

    def on_selection_changed(self, index):
        if 0 <= index < len(self._choices):
            self._selected = index
            truths, row = self.selected()
            self.describe(truths, row)
            self.style_crosses()
            self.draw_neutrino()

    def corrected(self):
        return getattr(self._module, 'correct_3d_objects', lambda: True)()

    def selected(self):
        ''' (truths, row) shown, or None '''
        if not self._choices:
            return None
        p, row = self._choices[min(self._selected, len(self._choices) - 1)]
        return self._data[p], row

    def drawObjects(self):
        selected = self.selected()
        if selected is None:
            # no MC truth in this event
            self.hide_dock()
            self.draw_neutrino()
            return
        self.describe(*selected)
        panel = self.panel()
        show = panel is None or panel.show_vertex.isChecked()
        self._crosses = []
        layout0 = ViewLayout(self._geom, plane=0)
        vertices = []
        for choice, (p, row) in enumerate(self._choices):
            truths = self._data[p]
            if not truths.has_neutrino[row]:
                continue
            vertex = Points(id=[row], x=[truths.vx[row]], y=[truths.vy[row]], z=[truths.vz[row]])
            correction = None
            if self.corrected():
                correction = point_correction(layout0, vertex, t0=mc_t0(self._geom, [truths.vt[row]]),
                                              tpc_source="readout window's")
            vertices.append((choice, vertex, correction, self.vertex_tooltip(truths, row, correction)))
        for view in self.views():
            layout = self.view_layout(view)
            items = []
            for choice, vertex, correction, tooltip in vertices:
                for _, (_, x, y) in project_objects(layout, vertex, correction).items():
                    for vx, vy in zip(x, y):
                        cross = CrossMarker(vx, vy)
                        cross.setToolTip(tooltip)
                        cross.setVisible(show)
                        items.append(cross)
                        self._crosses.append((choice, cross))
            self.add_to_view(view, items)
        self.style_crosses()
        self.draw_neutrino()

    def draw_neutrino(self):
        '''
        the shown interaction's neutrino annotated, if the dock says so: in
        each view, a dashed line to the vertex with an arrowhead, and its name
        '''
        remove_items(self._nu_items)
        self._nu_items = []
        panel, selected = self.panel(), self.selected()
        if panel is None or not panel.show_neutrino.isChecked() or selected is None \
                or not selected[0].has_neutrino[selected[1]]:
            self._set_note('neutrino', '')
            return
        truths, row = selected
        stored, producer = self.stored(truths), self._producer_of(truths)
        momentum = stored.nu_momentum(row)
        vertex = np.array([stored.vx[row], stored.vy[row], stored.vz[row]], dtype=float)
        path = neutrino_path(self._geom.detector, vertex, momentum)
        if path is None:
            self._set_note('neutrino', 'No neutrino direction in this file (the reader may predate it: rebuild it)'
                           if not np.all(np.isfinite(momentum)) else 'The neutrino vertex is outside the TPCs')
            return
        self._set_note('neutrino', '')
        start, end = path
        n = int(np.ceil(np.linalg.norm(end - start) / NEUTRINO_STEP_CM)) + 1
        xyz = start + np.linspace(0, 1, max(n, 2))[:, None] * (end - start)
        if producer is not None and self.space_charge_applied(producer):
            xyz = self._sce_map.displace(xyz)
        line = Tracks(id=[row], point_offset=[0, len(xyz)], x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                      valid=np.ones(len(xyz), dtype=bool))
        # where the arrow points: the last two points, projected as the line is
        tip = Tracks(id=[row], point_offset=[0, 2], x=xyz[-2:, 0], y=xyz[-2:, 1], z=xyz[-2:, 2],
                     valid=np.ones(2, dtype=bool))
        t0 = mc_t0(self._geom, float(truths.vt[row]))
        font_size = getattr(self._module, 'annotation_font_size', lambda: FONT_POINT_SIZE)()
        rotate = getattr(self._module, 'annotation_labels_rotated', lambda: True)() is not False
        for view in self.views():
            layout = self.view_layout(view)
            segments, tip_segments = (
                track_segments(layout, t, 0, track_correction(layout, t, 0, t0=t0) if self.corrected() else None)
                for t in (line, tip))
            segments = [np.stack([s.x, s.y], axis=1) for s in segments]
            if not segments:
                continue
            if tip_segments:
                # the vertex's segment last, for the arrowhead
                at = np.array([tip_segments[-1].x[-1], tip_segments[-1].y[-1]])
                segments.sort(key=lambda s: -np.linalg.norm(s[-1] - at))
            label_key = (producer, row, (view.plane(), view.cryostat()))
            items = annotation_items(segments, particle_name(truths.nu_pdg[row]), line_style='dashed',
                                     label_pos=self._nu_label_positions.get(label_key),
                                     on_moved=lambda pos, k=label_key: self._nu_label_positions.__setitem__(k, pos),
                                     font_size=font_size if isinstance(font_size, int) and font_size > 0
                                     else FONT_POINT_SIZE, rotate=rotate, arrow=True)
            add_items(view._view, items)
            self._nu_items.extend(items)

    def drawn_items(self):
        ''' the graphics items of the neutrino's annotation, in the views '''
        return list(self._nu_items)

    def style_crosses(self):
        ''' the selected interaction's vertex highlighted '''
        for choice, cross in self._crosses:
            cross.set_selected(choice == self._selected)

    def clearDrawnObjects(self, obj_list=None):
        super().clearDrawnObjects(obj_list)
        self._crosses = []
        remove_items(self._nu_items)
        self._nu_items = []

    def vertex_tooltip(self, truths, row, correction):
        what = correction.describe(0) if correction is not None else AS_STORED
        note = self.space_charge_note(truths)
        truths = self.stored(truths)
        return (f'{particle_name(truths.nu_pdg[row])} interaction vertex: ({truths.vx[row]:.1f}, {truths.vy[row]:.1f}, {truths.vz[row]:.1f}) cm, '
                f'{mc_t0(self._geom, truths.vt[row]) / 1e3:.2f} us from the trigger\n{what}' + note)

    def tpc_of(self, xyz):
        ''' (cryostat, TPC) containing a point, as "C:T" (or the TPC number with one cryostat); None outside '''
        detector = self._geom.detector
        cryostat, tpc = position_tpcs(detector, [xyz])[0]
        if not detector.tpc(cryostat, tpc).active_bounds.contains(xyz):
            return None
        return str(tpc) if len(detector.cryostats) == 1 else f'{cryostat}:{tpc}'

    def describe(self, truths, row):
        ''' fill the MC truth dock with an interaction '''
        panel = self.panel()
        if panel is None:
            return
        t0_us = mc_t0(self._geom, truths.vt[row]) / 1e3 if truths.has_neutrino[row] else None
        # the vertex where the interaction was, not where its charge was moved
        panel.set_summary(truth_summary(self.stored(truths), row, tpc_of=self.tpc_of, t0_us=t0_us))
        panel.set_has_vertices(any(self._data[p].has_neutrino[r] for p, r in self._choices))
        self.show_particles()
        self._module.show_mctruth_dock(True)

    def show_particles(self):
        panel, selected = self.panel(), self.selected()
        if panel is None or selected is None:
            return
        truths, row = selected
        # the same interaction listed again (e.g. all particles): its annotations stay
        same = self._listed is not None and self._listed[0] is truths and self._listed[1] == row
        annotated = panel.annotated_particles() if same else ()
        secondaries = panel.secondaries_particles() if same else ()
        paths = panel.paths_particles() if same else None
        self._listed = (truths, row)
        panel.set_particles(particle_rows(truths, row, final_state_only=not panel.all_particles.isChecked()),
                            annotated=annotated, secondaries=secondaries, paths=paths)

    def toggle_cross(self):
        show = self.panel().show_vertex.isChecked()
        for item in self._drawnObjects:
            item.setVisible(show)

    def select_particles(self, indices):
        ''' particles selected in the dock: their MCTracks highlighted, and a note on which '''
        selected, panel = self.selected(), self.panel()
        keys = [] if selected is None else [particle_key(*selected, i) for i in indices]
        select = getattr(self._module, 'select_mc_particles', None)
        tracks = None if select is None else select(keys)
        self._set_note('selected', self.track_note(keys, tracks))

    def on_particles_annotated(self, indices):
        '''
        particles annotated in the dock: their MCTracks drawn in white with
        their names, and those of their secondaries if asked; their paths
        drawn for the particles whose Tr box is checked
        '''
        selected, panel = self.selected(), self.panel()

        def keys_of(indices):
            return [] if selected is None else [particle_key(*selected, i) for i in indices]

        keys = keys_of(indices)
        secondaries = [] if panel is None else keys_of(panel.secondaries_particles())
        paths = None if panel is None else keys_of(panel.paths_particles())
        annotate = getattr(self._module, 'annotate_mc_particles', None)
        tracks = None if annotate is None else annotate(keys, secondaries, paths)
        note = ''
        if (keys or secondaries) and tracks is None:
            note = "No MCTracks (sim::MCTrack) in this file to annotate the particles with"
        elif keys:
            missing = sorted({particle_name(k.pdg) for k in keys if k not in tracks})
            note = (f'Nothing to annotate for {", ".join(missing)} (no trajectory, start or end '
                    f'known in the views shown)') if missing else ''
        self._set_note('annotated', note)

    def refresh_notes(self):
        '''
        the dock's note on the selected particles (and their highlights)
        again, e.g. once MCTracks are drawn or removed; annotations don't
        depend on them
        '''
        panel = self.panel()
        if panel is not None:
            self.select_particles(panel.selected_particles())

    def _set_note(self, which, text):
        self._notes[which] = text
        panel = self.panel()
        if panel is not None:
            panel.set_track_note('\n'.join(n for n in self._notes.values() if n))

    @staticmethod
    def track_note(keys, tracks):
        '''
        what the dock says about the selected particles' MCTracks. tracks:
        {key: [MCTrack IDs]} found, or None if no MCTracks are drawn
        '''
        if not keys:
            return ''
        if tracks is None:
            return 'Turn on MCTrack in the Products to highlight the selected particles\' tracks'
        found = [f'{particle_name(k.pdg)}: MCTrack {", ".join(str(t) for t in tracks[k])}' for k in keys if k in tracks]
        missing = sorted({particle_name(k.pdg) for k in keys if k not in tracks})
        lines = found
        if missing:
            lines = lines + [f'No MCTrack for {", ".join(missing)} (neutral, or not tracked by mcreco)']
        return '\n'.join(lines)
