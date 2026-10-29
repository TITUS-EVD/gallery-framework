from titus.display import ViewLayout, mc_t0, point_correction, position_tpcs, project_objects
from titus.display.mctruth import particle_rows, truth_label, truth_summary
from titus.display.projection import AS_STORED
from titus.drawables.drawable import ReaderDrawable
from titus.drawables.items import CrossMarker
from titus.readers import Points


class MCTruth(ReaderDrawable):
    '''
    simb::MCTruth: one interaction at a time, chosen in the MC truth dock
    (neutrinos first). The dock (titus.gui.mctruth_panel, in the TPC module)
    describes it and lists its particles, and the neutrino vertex is drawn as
    a cross in the wire views. Corrected (the TPC module's "Correct 3D
    objects" option), the interaction time is undone like a T0, so the
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
        panel = self.panel()
        if panel is not None:
            panel.dropdown.currentIndexChanged.connect(self.on_selection_changed)
            panel.show_vertex.stateChanged.connect(self.toggle_cross)
            panel.all_particles.toggled.connect(lambda _: self.show_particles())

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
            self.clearDrawnObjects()
            self.drawObjects()

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
        truths, row = selected
        self.describe(truths, row)
        if not truths.has_neutrino[row]:
            return
        vertex = Points(id=[row], x=[truths.vx[row]], y=[truths.vy[row]], z=[truths.vz[row]])
        correction = None
        if self.corrected():
            correction = point_correction(ViewLayout(self._geom, plane=0), vertex, t0=mc_t0(self._geom, [truths.vt[row]]),
                                          tpc_source="readout window's")
        tooltip = self.vertex_tooltip(truths, row, correction)
        panel = self.panel()
        show = panel is None or panel.show_vertex.isChecked()
        for view in self.views():
            items = []
            for _, (_, x, y) in project_objects(self.view_layout(view), vertex, correction).items():
                for vx, vy in zip(x, y):
                    cross = CrossMarker(vx, vy)
                    cross.setToolTip(tooltip)
                    cross.setVisible(show)
                    items.append(cross)
            self.add_to_view(view, items)

    def vertex_tooltip(self, truths, row, correction):
        what = correction.describe(0) if correction is not None else AS_STORED
        return (f'Neutrino interaction vertex: ({truths.vx[row]:.1f}, {truths.vy[row]:.1f}, {truths.vz[row]:.1f}) cm, '
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
        panel.set_has_vertex(bool(truths.has_neutrino[row]))
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
