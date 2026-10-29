from titus.display import ViewLayout, mc_t0, point_correction, project_objects
from titus.display.projection import AS_STORED
from titus.drawables.drawable import ReaderDrawable
from titus.drawables.items import CrossMarker
from titus.readers import Points

# Maximum number of final state particles listed
MAX_FS_PARS = 6

pdg_to_name = {
    12: 'nue',
    -12: 'anue',
    14: 'numu',
    -14: 'numubar',
}

mode_to_name = {
    0: 'QE',
    1: 'RES',
    2: 'DIS',
    3: 'COH',
    10: 'MEC',
}

inttype_to_name = {
    0: 'CC',
    1: 'NC',
}

# simb::Origin_t
ORIGIN_NAMES = {0: 'Unknown', 1: 'Neutrino', 2: 'Cosmic', 3: 'Supernova', 4: 'Single particle'}


class MCTruth(ReaderDrawable):
    '''
    simb::MCTruth: one interaction at a time, chosen in the MC truth dock
    (neutrinos first). The dock describes it, and the neutrino vertex is
    drawn as a cross in the wire views. Corrected (the TPC module's "Correct
    3D objects" option), the interaction time is undone like a T0, so the
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
        dropdown = getattr(tpc_module, '_mctruth_dropdown', None)
        if dropdown is not None:
            dropdown.currentIndexChanged.connect(self.on_selection_changed)
        show = getattr(tpc_module, '_show_vertex', None)
        if show is not None:
            show.stateChanged.connect(self.toggle_cross)

    def on_data(self):
        self._choices = sorted(((p, row) for p, truths in enumerate(self._data) for row in range(len(truths))),
                               key=lambda c: not self._data[c[0]].has_neutrino[c[1]])
        self._selected = 0
        dropdown = self._module._mctruth_dropdown
        dropdown.blockSignals(True)
        dropdown.clear()
        for p, row in self._choices:
            truths = self._data[p]
            origin = ORIGIN_NAMES.get(int(truths.origin[row]), str(truths.origin[row]))
            dropdown.addItem(f'{origin} ({str(self.producers()[p]).split(":")[0]} {row + 1})')
        dropdown.blockSignals(False)

    def on_selection_changed(self, index):
        if 0 <= index < len(self._choices):
            self._selected = index
            self.clearDrawnObjects(hide=False)
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
        show = getattr(self._module, '_show_vertex', None)
        for view in self.views():
            items = []
            for _, (_, x, y) in project_objects(self.view_layout(view), vertex, correction).items():
                for vx, vy in zip(x, y):
                    cross = CrossMarker(vx, vy)
                    cross.setToolTip(tooltip)
                    cross.setVisible(show is None or show.isChecked())
                    items.append(cross)
            self.add_to_view(view, items)

    def vertex_tooltip(self, truths, row, correction):
        what = correction.describe(0) if correction is not None else AS_STORED
        return (f'Neutrino interaction vertex: ({truths.vx[row]:.1f}, {truths.vy[row]:.1f}, {truths.vz[row]:.1f}) cm, '
                f'{mc_t0(self._geom, truths.vt[row]) / 1e3:.2f} us from the trigger\n{what}')

    def describe(self, truths, row):
        ''' the dock's text for an interaction '''
        message, tooltip = '', ''
        pdg, status, energy = truths.particles(row)
        final = status == 1
        fs_pdgs, fs_enes = pdg[final], energy[final]
        origin = int(truths.origin[row])
        if origin == 1 and truths.has_neutrino[row]:
            nu_pdg = int(truths.nu_pdg[row])
            name = pdg_to_name.get(nu_pdg, 'NA')
            mode = mode_to_name.get(int(truths.mode[row]), 'NA')
            inttype = inttype_to_name.get(int(truths.ccnc[row]), 'NA')
            message = f'PDG: {nu_pdg} ({name}), {inttype}, E: {truths.nu_energy[row]:.3} GeV, mode: {mode}'
            tooltip = (f'Vertex: x = {truths.vx[row]:.2f}, y = {truths.vy[row]:.2f}, z = {truths.vz[row]:.2f}\n'
                       'Final State Particles:')
            for p, e in zip(fs_pdgs[:MAX_FS_PARS], fs_enes[:MAX_FS_PARS]):
                tooltip += f'\n  PDG: {p}, Energy: {e:.3} GeV'
            if len(fs_pdgs) > MAX_FS_PARS:
                tooltip += '\n  ... [some particles not shown]'
        elif origin == 2:
            message = f'Cosmic origin, {len(fs_pdgs)} particles'
        elif origin == 3:
            message = 'Supernova event'
        elif origin == 4:
            if len(fs_pdgs) == 1:
                message = f'Single particle. PDG: {fs_pdgs[0]}, Energy: {fs_enes[0]:.3} GeV.'
            else:
                message = f'Particle gun with {len(fs_pdgs)} particles'
                tooltip = 'All Particles:' + ''.join(f'\n  PDG: {p}, Energy: {e:.3} GeV'
                                                     for p, e in zip(fs_pdgs, fs_enes))
        module = self._module
        module._mctruth_text1.setText(message)
        module._mctruth_text1.setToolTip(tooltip)
        module._mctruth_text2.setText(tooltip)
        module._mctruth_dock.show()

    def toggle_cross(self):
        show = self._module._show_vertex.isChecked()
        for item in self._drawnObjects:
            item.setVisible(show)

    def clearDrawnObjects(self, obj_list=None, hide=True):
        super().clearDrawnObjects(obj_list)
        if hide and getattr(self._module, '_mctruth_dock', None) is not None:
            self._module._mctruth_dock.hide()
