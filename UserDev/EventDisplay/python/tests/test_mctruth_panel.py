"""
The MC truth dock's particle table: its Annotate boxes, independent of the
row selection (which highlights the particles' MCTracks).
"""

import pytest

pytest.importorskip('PyQt5')
from PyQt5 import QtCore, QtTest  # noqa: E402

from titus.display.mctruth import ParticleRow  # noqa: E402
from titus.gui.mctruth_panel import MCTruthPanel  # noqa: E402

ROWS = [ParticleRow('μ⁻', 13, 2119., 2.225, 'final', 0), ParticleRow('p', 2212, 312., 1.25, 'final', 1),
        ParticleRow('γ', 22, 50., 0.05, 'final', 2), ParticleRow('K⁺', 321, 512., 1.006, 'final', 4)]


@pytest.fixture
def panel(qtbot):
    panel = MCTruthPanel()
    qtbot.addWidget(panel)
    panel.resize(500, 400)
    panel.set_particles(ROWS)
    panel.show()
    qtbot.waitExposed(panel)
    return panel


def click(panel, index, column):
    ''' click a particle's cell: an Annotate box on its check indicator '''
    table = panel._table
    row = next(r for r in range(table.rowCount()) if panel._particle_index(r) == index)
    rect = table.visualItemRect(table.item(row, panel.COLUMNS.index(column)))
    pos = QtCore.QPoint(rect.left() + 10, rect.center().y()) if column == 'Annotate' else rect.center()
    QtTest.QTest.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=pos)


def test_annotate_boxes_are_not_the_selection(panel):
    annotated, selected = [], []
    panel.particlesAnnotated.connect(annotated.append)
    panel.particlesSelected.connect(selected.append)
    assert panel._table.horizontalHeaderItem(0).text() == '✎'
    click(panel, 0, 'Annotate')
    assert panel.annotated_particles() == [0] and panel.selected_particles() == []
    click(panel, 1, 'Particle')
    assert panel.annotated_particles() == [0] and panel.selected_particles() == [1]
    click(panel, 4, 'Annotate')
    assert panel.annotated_particles() == [0, 4] and panel.selected_particles() == [1]
    click(panel, 0, 'Annotate')
    assert annotated == [[0], [0, 4], [4]] and selected == [[1]]


def test_showers_are_not_annotated_yet(panel):
    click(panel, 2, 'Annotate')
    assert panel.annotated_particles() == []
    panel.annotate_particles([2, 4])
    assert panel.annotated_particles() == [4]
    row = next(r for r in range(panel.particle_count()) if panel.cell(r, 'Particle') == 'γ')
    assert 'showers' in panel._table.item(row, 0).toolTip()


def test_listed_again_annotations_kept_if_asked(panel):
    panel.annotate_particles([0, 1])
    emitted = []
    panel.particlesAnnotated.connect(emitted.append)
    # the same interaction (e.g. all its particles): kept; another: none
    panel.set_particles(ROWS, annotated=panel.annotated_particles())
    assert panel.annotated_particles() == [0, 1] and emitted == [[0, 1]]
    panel.set_particles(ROWS)
    assert panel.annotated_particles() == [] and emitted[-1] == []
    # the other columns where they were, by name
    assert {panel.cell(r, 'PDG') for r in range(panel.particle_count())} == {'13', '2212', '22', '321'}
