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


class MCTruthPanel(QtWidgets.QWidget):
    '''
    dropdown: the interaction shown (filled by the MCTruth drawable);
    show_vertex: draw the neutrino vertex; all_particles: list every
    particle of the generator's record, not only the final state.
    '''
    COLUMNS = ('Particle', 'PDG', 'KE [MeV]', 'E [GeV]', 'Status')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.setToolTip('MC truth interactions of this event (neutrinos first)')

        # one row per summary line: bold label, value wrapped to the dock's width
        self._summary = QtWidgets.QWidget()
        self._summary_layout = QtWidgets.QFormLayout(self._summary)
        self._summary_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_layout.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)

        self.all_particles = QtWidgets.QCheckBox('All particles')
        self.all_particles.setToolTip('Also list the initial, intermediate and in-nucleus particles '
                                      '(GENIE status codes), not only the final state')

        self._table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setDefaultSectionSize(self.fontMetrics().height() + 6)
        self._table.setColumnHidden(self.COLUMNS.index('Status'), True)
        self.all_particles.toggled.connect(self._show_status_column)
        # most energetic first
        self._table.sortItems(self.COLUMNS.index('KE [MeV]'), QtCore.Qt.DescendingOrder)

        self.show_vertex = QtWidgets.QCheckBox('Show neutrino vertex')
        self.show_vertex.setToolTip('Mark the neutrino interaction vertex with a cross in the wire views')
        self.show_vertex.setChecked(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.dropdown)
        layout.addWidget(self._summary)
        options = QtWidgets.QHBoxLayout()
        options.addWidget(self.show_vertex)
        options.addStretch()
        options.addWidget(self.all_particles)
        layout.addLayout(options)
        layout.addWidget(self._table, stretch=1)
        self.clear()

    def _show_status_column(self, show):
        self._table.setColumnHidden(self.COLUMNS.index('Status'), not show)
        self._fit_columns()

    def clear(self):
        self.dropdown.blockSignals(True)
        self.dropdown.clear()
        self.dropdown.blockSignals(False)
        self.set_summary([('', 'No MC truth selected')])
        self.set_particles([])

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

    def set_particles(self, rows):
        ''' ParticleRows (titus.display.mctruth.particle_rows) '''
        table = self._table
        sort_column = table.horizontalHeader().sortIndicatorSection()
        order = table.horizontalHeader().sortIndicatorOrder()
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            name = QtWidgets.QTableWidgetItem(row.name)
            if is_genie_pseudo_particle(row.pdg):
                name.setToolTip('GENIE pseudo-particle (e.g. nucleon cluster, hadronic system), no mass')
            table.setItem(i, 0, name)
            table.setItem(i, 1, _NumberItem(row.pdg, '{:d}'))
            table.setItem(i, 2, _NumberItem(row.kinetic_energy, '{:.1f}'))
            table.setItem(i, 3, _NumberItem(row.energy, '{:.3f}'))
            table.setItem(i, 4, QtWidgets.QTableWidgetItem(row.status))
        table.setSortingEnabled(True)
        table.sortItems(sort_column, order)
        self._fit_columns()

    def _fit_columns(self):
        table = self._table
        table.resizeColumnsToContents()
        # wide enough for every column, so the dock doesn't hide the last ones
        header = table.horizontalHeader()
        # (from each column's contents: the last section is stretched to the table's width)
        width = sum(max(table.sizeHintForColumn(c), header.sectionSizeHint(c))
                    for c in range(table.columnCount()) if not table.isColumnHidden(c))
        table.setMinimumWidth(width + table.verticalScrollBar().sizeHint().width() + 2 * table.frameWidth())

    def set_has_vertex(self, has_vertex):
        ''' the vertex checkbox only means something for neutrino interactions '''
        self.show_vertex.setEnabled(has_vertex)

    def particle_count(self):
        return self._table.rowCount()

    def cell(self, row, column):
        item = self._table.item(row, self.COLUMNS.index(column))
        return None if item is None else item.text()
