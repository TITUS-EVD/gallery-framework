"""
The MC truth dock's contents: which interaction is shown, its summary and
its particles (titus.display.mctruth).
"""

from PyQt5 import QtCore, QtWidgets

from titus.display.mctruth import is_genie_pseudo_particle


class _NumberItem(QtWidgets.QTableWidgetItem):
    ''' a table cell sorted by its number; blank without one (sorted first) '''

    def __init__(self, value, fmt):
        super().__init__('' if value is None else fmt.format(value))
        self._value = float('-inf') if value is None else value
        self.setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))

    def __lt__(self, other):
        if isinstance(other, _NumberItem):
            return self._value < other._value
        return super().__lt__(other)


class _CheckItem(QtWidgets.QTableWidgetItem):
    ''' an annotate checkbox, sorted checked first (in descending order) '''

    def __lt__(self, other):
        return self.checkState() < other.checkState() if isinstance(other, _CheckItem) else super().__lt__(other)


class _AllCheckBox(QtWidgets.QCheckBox):
    ''' the checkbox of a column of boxes: partly checked when they differ; a click checks or unchecks them all '''

    def nextCheckState(self):
        self.setCheckState(QtCore.Qt.Unchecked if self.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked)


# particles drawn as showers (sim::MCShower, no MCTrack to annotate): electrons, photons, neutral pions
SHOWERING_PDG = (11, 22, 111)


class MCTruthPanel(QtWidgets.QWidget):
    '''
    dropdown: the interaction shown (filled by the MCTruth drawable);
    show_vertex: draw the neutrino vertices; all_particles: list every
    particle of the generator's record, not only the final state.
    particlesSelected: the table's selected particles, as indices into the
    interaction's particles; particlesAnnotated: those whose Annotate box is
    checked (drawn with their name), the same way, emitted too when a
    Secondaries box changes (secondaries_particles: those whose descendants
    are annotated) or a Paths box (paths_particles: those whose annotations,
    the particle's and its secondaries', are drawn with their paths, else
    names only). show_paths checks or unchecks all the Paths boxes.
    '''
    COLUMNS = ('Annotate', 'Secondaries', 'Paths', 'Particle', 'PDG', 'KE [MeV]', 'E [GeV]', 'Status')
    HEADERS = {'Annotate': '✎', 'Secondaries': '2°', 'Paths': 'Tr'}
    # the checkbox columns
    CHECKS = ('Annotate', 'Secondaries', 'Paths')
    TOOLTIPS = {'Annotate': 'Annotate the {}: drawn in white, with its name',
                'Secondaries': 'Annotate what the {} made, each with its name',
                'Paths': "Draw the {}'s annotated paths (its own and its secondaries'); unchecked, only names"}
    particlesSelected = QtCore.pyqtSignal(list)
    particlesAnnotated = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.setToolTip('MC truth interactions of this event (neutrinos first)')

        # one row per summary line: bold label, value wrapped to the dock's width
        self._summary = QtWidgets.QWidget()
        self._summary_layout = QtWidgets.QFormLayout(self._summary)
        self._summary_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_layout.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)

        self.show_paths = _AllCheckBox('Show paths')
        self.show_paths.setChecked(True)
        self.show_paths.setToolTip("Draw the annotated particles' paths in white; unchecked, only their names.\n"
                                   "Sets every particle's Tr box (partly checked: some are off)")
        self.show_paths.clicked.connect(lambda: self.show_all_paths(self.show_paths.checkState() == QtCore.Qt.Checked))
        self.all_particles = QtWidgets.QCheckBox('All particles')
        self.all_particles.setToolTip('Also list the initial, intermediate and in-nucleus particles '
                                      '(GENIE status codes), not only the final state')

        self._table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels([self.HEADERS.get(c, c) for c in self.COLUMNS])
        self._table.horizontalHeaderItem(self._column('Annotate')).setToolTip(
            "Annotate: draw the particle's MCTrack in white, with its name (dashed and straight\n"
            "from its start to its end without a trajectory, e.g. neutral particles)")
        self._table.horizontalHeaderItem(self._column('Secondaries')).setToolTip(
            "Secondaries: annotate what the particle made (decays, inelastic scatters...: the MCTracks\n"
            "descending from it in Geant4), each with its name")
        self._table.horizontalHeaderItem(self._column('Paths')).setToolTip(
            "Tr: draw the paths of the particle's annotations (its own and its secondaries');\n"
            "unchecked, only their names (e.g. only the dashed lines of neutral particles)")
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.setToolTip('Select particles to highlight their MCTracks (turn on MCTrack in the Products)')
        self._table.itemSelectionChanged.connect(
            lambda: self.particlesSelected.emit(self.selected_particles()))
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setDefaultSectionSize(self.fontMetrics().height() + 6)
        self._table.setColumnHidden(self.COLUMNS.index('Status'), True)
        self.all_particles.toggled.connect(self._show_status_column)
        # most energetic first
        self._table.sortItems(self.COLUMNS.index('KE [MeV]'), QtCore.Qt.DescendingOrder)

        self.show_vertex = QtWidgets.QCheckBox('Show neutrino vertices')
        self.show_vertex.setToolTip('Mark the neutrino interaction vertices with crosses in the wire views; '
                                    'the selected interaction\'s is highlighted')
        # off unless asked (the TPC module keeps the choice between sessions)
        self.show_vertex.setChecked(False)

        # which MCTracks the selected particles are
        self._track_note = QtWidgets.QLabel()
        self._track_note.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        # for every interaction's vertex: above the one shown
        layout.addWidget(self.show_vertex)
        layout.addWidget(self.dropdown)
        layout.addWidget(self._summary)
        options = QtWidgets.QHBoxLayout()
        options.addWidget(self.show_paths)
        options.addStretch()
        options.addWidget(self.all_particles)
        layout.addLayout(options)
        layout.addWidget(self._table, stretch=1)
        layout.addWidget(self._track_note)
        self.clear()

    def _column(self, name):
        return self.COLUMNS.index(name)

    def _show_status_column(self, show):
        self._table.setColumnHidden(self.COLUMNS.index('Status'), not show)
        self._fit_columns()

    def _on_item_changed(self, item):
        if item.column() in [self._column(c) for c in self.CHECKS]:
            if item.column() == self._column('Paths'):
                self._show_paths_state()
            self.particlesAnnotated.emit(self.annotated_particles())

    def _show_paths_state(self):
        ''' Show paths as the Tr boxes are: checked, unchecked, or partly checked if they differ '''
        boxes = [self._table.item(r, self._column('Paths')) for r in range(self._table.rowCount())]
        states = {b.checkState() for b in boxes if b.flags() & QtCore.Qt.ItemIsEnabled}
        if states:
            self.show_paths.setCheckState(states.pop() if len(states) == 1 else QtCore.Qt.PartiallyChecked)

    def show_all_paths(self, show):
        ''' check (or uncheck) every Tr box: the annotations drawn with their paths, or names only '''
        indices = [self._particle_index(r) for r in range(self._table.rowCount())] if show else []
        self.annotate_particles(self.annotated_particles(), paths=indices)
        self.show_paths.setCheckState(QtCore.Qt.Checked if show else QtCore.Qt.Unchecked)

    def clear(self):
        self.dropdown.blockSignals(True)
        self.dropdown.clear()
        self.dropdown.blockSignals(False)
        self.set_summary([('', 'No MC truth selected')])
        self.set_particles([])
        self.set_track_note('')

    def set_summary(self, lines):
        ''' (label, value) lines, as from titus.display.mctruth.truth_summary '''
        while self._summary_layout.rowCount():
            self._summary_layout.removeRow(0)
        for label, value in lines:
            name = QtWidgets.QLabel(label)
            font = name.font()
            font.setBold(True)
            name.setFont(font)
            text = QtWidgets.QLabel(value)
            text.setWordWrap(True)
            text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self._summary_layout.addRow(name, text)

    def summary(self):
        ''' the summary shown, as (label, value) lines '''
        layout = self._summary_layout
        return [(layout.itemAt(r, QtWidgets.QFormLayout.LabelRole).widget().text(),
                 layout.itemAt(r, QtWidgets.QFormLayout.FieldRole).widget().text())
                for r in range(layout.rowCount())]

    def set_particles(self, rows, annotated=(), secondaries=(), paths=None):
        '''
        ParticleRows (titus.display.mctruth.particle_rows). annotated,
        secondaries, paths: indices of particles to keep annotated, with
        their secondaries, with their paths (the same interaction listed
        again); paths None: every Tr box as Show paths says (checked unless
        unchecked)
        '''
        if paths is None:
            paths = [] if self.show_paths.checkState() == QtCore.Qt.Unchecked else [row.index for row in rows]
        table = self._table
        sort_column = table.horizontalHeader().sortIndicatorSection()
        order = table.horizontalHeader().sortIndicatorOrder()
        table.blockSignals(True)
        table.clearSelection()
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            name = QtWidgets.QTableWidgetItem(row.name)
            name.setData(QtCore.Qt.UserRole, row.index)
            if is_genie_pseudo_particle(row.pdg):
                name.setToolTip('GENIE pseudo-particle (e.g. nucleon cluster, hadronic system), no mass')
            for column, checked in (('Annotate', annotated), ('Secondaries', secondaries), ('Paths', paths)):
                table.setItem(i, self._column(column), self._annotate_item(row, row.index in checked, column))
            table.setItem(i, self._column('Particle'), name)
            table.setItem(i, self._column('PDG'), _NumberItem(row.pdg, '{:d}'))
            table.setItem(i, self._column('KE [MeV]'), _NumberItem(row.kinetic_energy, '{:.1f}'))
            table.setItem(i, self._column('E [GeV]'), _NumberItem(row.energy, '{:.3f}'))
            table.setItem(i, self._column('Status'), QtWidgets.QTableWidgetItem(row.status))
        table.setSortingEnabled(True)
        table.sortItems(sort_column, order)
        table.blockSignals(False)
        self._fit_columns()
        self._show_paths_state()
        # a new list: nothing selected; annotated as asked
        self.particlesSelected.emit([])
        self.particlesAnnotated.emit(self.annotated_particles())

    @classmethod
    def _annotate_item(cls, row, checked, column='Annotate'):
        '''
        the Annotate, Secondaries or Paths box of a particle: not
        selectable, so that checking it leaves the selection (highlight) as
        it is. Showering particles have no MCTrack to annotate yet.
        '''
        item = _CheckItem()
        if abs(row.pdg) in SHOWERING_PDG:
            item.setFlags(QtCore.Qt.ItemIsUserCheckable)
            item.setToolTip(f'{row.name}: showers are not annotated yet')
        else:
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setToolTip(cls.TOOLTIPS[column].format(row.name))
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        return item

    def _particle_index(self, row):
        return self._table.item(row, self._column('Particle')).data(QtCore.Qt.UserRole)

    def selected_particles(self):
        ''' indices (into the interaction's particles) of the selected rows '''
        rows = sorted({item.row() for item in self._table.selectedItems()})
        return [self._particle_index(r) for r in rows]

    def _checked(self, name):
        ''' the particles whose box of this column is checked (showers' boxes, disabled, left out) '''
        items = [(r, self._table.item(r, self._column(name))) for r in range(self._table.rowCount())]
        return sorted(self._particle_index(r) for r, item in items
                      if item.checkState() == QtCore.Qt.Checked and item.flags() & QtCore.Qt.ItemIsEnabled)

    def annotated_particles(self):
        ''' indices (into the interaction's particles) of the particles whose Annotate box is checked '''
        return self._checked('Annotate')

    def secondaries_particles(self):
        ''' indices (into the interaction's particles) of the particles whose Secondaries box is checked '''
        return self._checked('Secondaries')

    def paths_particles(self):
        ''' indices (into the interaction's particles) of the particles whose Tr (Paths) box is checked '''
        return self._checked('Paths')

    def annotate_particles(self, indices, secondaries=None, paths=None):
        '''
        check the Annotate boxes of these particles, uncheck the others; the
        same for the Secondaries and Paths boxes, unless None (e.g. tests)
        '''
        table = self._table
        table.blockSignals(True)
        for name, checked in (('Annotate', indices), ('Secondaries', secondaries), ('Paths', paths)):
            if checked is None:
                continue
            for r in range(table.rowCount()):
                item = table.item(r, self._column(name))
                if item.flags() & QtCore.Qt.ItemIsEnabled:
                    item.setCheckState(QtCore.Qt.Checked if self._particle_index(r) in checked
                                       else QtCore.Qt.Unchecked)
        table.blockSignals(False)
        if paths is not None:
            self._show_paths_state()
        self.particlesAnnotated.emit(self.annotated_particles())

    def select_particles(self, indices):
        ''' select the rows of these particles (e.g. tests) '''
        table = self._table
        table.blockSignals(True)
        table.clearSelection()
        mode = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        for r in range(table.rowCount()):
            if self._particle_index(r) in indices:
                table.selectionModel().select(table.model().index(r, self._column('Particle')), mode)
        table.blockSignals(False)
        self.particlesSelected.emit(self.selected_particles())

    def set_track_note(self, text):
        self._track_note.setText(text)
        self._track_note.setVisible(bool(text))

    def track_note(self):
        return self._track_note.text() if self._track_note.isVisibleTo(self) else ''

    def _fit_columns(self):
        table = self._table
        table.resizeColumnsToContents()
        # wide enough for every column, so the dock doesn't hide the last ones
        header = table.horizontalHeader()
        # (from each column's contents: the last section is stretched to the table's width)
        width = sum(max(table.sizeHintForColumn(c), header.sectionSizeHint(c))
                    for c in range(table.columnCount()) if not table.isColumnHidden(c))
        table.setMinimumWidth(width + table.verticalScrollBar().sizeHint().width() + 2 * table.frameWidth())

    def set_has_vertices(self, has_vertices):
        ''' the vertex toggle only means something with neutrino interactions in the event '''
        self.show_vertex.setEnabled(has_vertices)

    def particle_count(self):
        return self._table.rowCount()

    def cell(self, row, column):
        item = self._table.item(row, self.COLUMNS.index(column))
        return None if item is None else item.text()
