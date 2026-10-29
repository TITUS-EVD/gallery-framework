from titus.display import ViewLayout, mc_t0, point_correction, position_tpcs, project_objects
from titus.display.mctruth import particle_key, particle_name, particle_rows, truth_label, truth_summary
from titus.display.projection import AS_STORED
from titus.drawables.drawable import ReaderDrawable
from titus.drawables.items import CrossMarker
from titus.readers import Points


class MCTruth(ReaderDrawable):
    '''
    simb::MCTruth: the MC truth dock (titus.gui.mctruth_panel, in the TPC
    module) describes one interaction at a time, chosen in its dropdown
    (neutrinos first), and lists its particles. Every neutrino vertex is
    drawn as a cross in the wire views, the selected interaction's
    highlighted. Selecting particles in the dock highlights their MCTracks
    (TpcModule.select_mc_particles). Corrected (the TPC module's "Correct 3D
    objects" option), each interaction's time is undone like a T0, so its
    vertex lands where its charge was read out.
    '''
    KIND = 'mctruths'

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
        panel = self.panel()
        if panel is not None:
            panel.dropdown.currentIndexChanged.connect(self.on_selection_changed)
            panel.show_vertex.stateChanged.connect(self.toggle_cross)
            panel.all_particles.toggled.connect(lambda _: self.show_particles())
            panel.particlesSelected.connect(self.select_particles)

    def panel(self):
        ''' the TPC module's MC truth panel (None in tests without one) '''
        return getattr(self._module, '_mctruth_panel', None)

    def on_data(self):
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

    def style_crosses(self):
        ''' the selected interaction's vertex highlighted '''
        for choice, cross in self._crosses:
            cross.set_selected(choice == self._selected)

    def clearDrawnObjects(self, obj_list=None):
        super().clearDrawnObjects(obj_list)
        self._crosses = []

    def vertex_tooltip(self, truths, row, correction):
        what = correction.describe(0) if correction is not None else AS_STORED
        return (f'{particle_name(truths.nu_pdg[row])} interaction vertex: ({truths.vx[row]:.1f}, {truths.vy[row]:.1f}, {truths.vz[row]:.1f}) cm, '
                f'{mc_t0(self._geom, truths.vt[row]) / 1e3:.2f} us from the trigger\n{what}')

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
        panel.set_summary(truth_summary(truths, row, tpc_of=self.tpc_of, t0_us=t0_us))
        panel.set_has_vertices(any(self._data[p].has_neutrino[r] for p, r in self._choices))
        self.show_particles()
        self._module.show_mctruth_dock(True)

    def show_particles(self):
        panel, selected = self.panel(), self.selected()
        if panel is None or selected is None:
            return
        truths, row = selected
        panel.set_particles(particle_rows(truths, row, final_state_only=not panel.all_particles.isChecked()))

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
        if panel is not None:
            panel.set_track_note(self.track_note(keys, tracks))

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
