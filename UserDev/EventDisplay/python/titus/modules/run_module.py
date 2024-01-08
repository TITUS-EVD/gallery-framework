#!/usr/bin/env python3

"""
This module adds run controls for gallery events: forward, back, goto, etc.
"""

from PyQt5 import QtWidgets, QtGui, QtCore

from .module import Module


class RunModule(Module):
    def __init__(self):
        super().__init__()

    def _initialize(self):
        """ Add run controls to the GUI in a DockWidget """

        self._event_dock =  QtWidgets.QDockWidget('Event Controls', self._gui)
        self._event_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._event_dock.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        frame = QtWidgets.QWidget(self._event_dock)
        main_layout = QtWidgets.QVBoxLayout()
        frame.setLayout(main_layout)
        self._event_dock.setWidget(frame)

        # this connection allows dock widget to be restored with activate function
        self._dock_widgets.append(self._event_dock)

        # Run display labels, horizontally centered
        self._run_label = QtWidgets.QLabel("Run: 0")
        self._event_label = QtWidgets.QLabel("Event: 0")
        self._subrun_label = QtWidgets.QLabel("Subrun: 0")
        run_control_layout = QtWidgets.QHBoxLayout()
        run_control_layout.addStretch()
        run_control_layout.addWidget(self._event_label)
        run_control_layout.addWidget(self._run_label)
        run_control_layout.addWidget(self._subrun_label)
        run_control_layout.addStretch()
        main_layout.addLayout(run_control_layout)

        # Next/Previous buttons
        self._prev_button = QtWidgets.QPushButton("Previous")
        self._prev_button.clicked.connect(self._gi.prev)
        self._prev_button.setToolTip("Move to the previous event.")
        self._next_button = QtWidgets.QPushButton("Next")
        self._next_button.clicked.connect(self._gi.next)
        self._next_button.setToolTip("Move to the next event.")
        prev_next_layout = QtWidgets.QHBoxLayout()
        prev_next_layout.addWidget(self._prev_button)
        prev_next_layout.addWidget(self._next_button)
        main_layout.addLayout(prev_next_layout)

        # Go to event
        self._larlite_event_entry = QtWidgets.QLineEdit()
        self._larlite_event_entry.setToolTip("Enter an event to skip to that event.")
        self._larlite_event_entry.returnPressed.connect(self.go_to_event)
        event_layout = QtWidgets.QHBoxLayout()
        event_layout.addWidget(QtWidgets.QLabel('Go to:'))
        event_layout.addWidget(self._larlite_event_entry)
        main_layout.addLayout(event_layout)

        self._run_entry = QtWidgets.QLineEdit()
        self._run_entry.setPlaceholderText('Run')
        self._subrun_entry = QtWidgets.QLineEdit()
        self._subrun_entry.setPlaceholderText('Subrun')
        self._event_entry = QtWidgets.QLineEdit()
        self._event_entry.setPlaceholderText('Event')
        self._goto_button = QtWidgets.QPushButton('Go')
        run_entry_layout = QtWidgets.QHBoxLayout()
        run_entry_layout.addWidget(self._run_entry)
        run_entry_layout.addWidget(self._subrun_entry)
        run_entry_layout.addWidget(self._event_entry)
        run_entry_layout.addWidget(self._goto_button)
        main_layout.addLayout(run_entry_layout)

        main_layout.addStretch()
        self._gui.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._event_dock)
        self._gui.resizeDocks([self._event_dock], [250], QtCore.Qt.Horizontal)

        # add some file controls in their expected places
        run_menu = QtWidgets.QMenu("&Run", self._gui)
        self._gui.menuBar().addMenu(run_menu)
        

        restore_action = QtWidgets.QAction('Restore controls...', run_menu)        
        restore_action.triggered.connect(self._on_restore_action)
        run_menu.addSeparator()
        run_menu.addAction(restore_action)
        self.update()

    def go_to_event(self):
        """ helper function for parsing text box """
        try:
            event = int(self._larlite_event_entry.text())
        except:
            print("Error, must enter an integer")
            self._larlite_event_entry.setText(str(self._event_manager.event()))
            return
        
        self._gi.go_to_event(event)

    def update(self):
        ''' Sets the text boxes correctly '''
        self._larlite_event_entry.setText(str(self._gi.event()))

        self._event_label.setText(f'Event: {self._gi.event()}')
        self._run_label.setText(f'Run: {self._gi.run()}')
        self._subrun_label.setText(f'Subrun: {self._gi.subrun()}')
        self.setupEventRunSubrun()

    def setupEventRunSubrun(self):
        self._run_entry.setMinimumWidth(40)
        self._subrun_entry.setMinimumWidth(40)
        self._event_entry.setMinimumWidth(40)
        runs = self._gi.available_runs()
        subruns = self._gi.available_subruns()
        events = self._gi.available_events()

        if len(runs) == 1:
            self._run_entry.setText(str(runs[0]))
            self._run_entry.setDisabled(True)
        else:
            tooltip_text = 'Available runs: '
            tooltip_text += ', '.join(map(str, runs))
            self._run_entry.setToolTip(tooltip_text)

        if len(subruns) == 1:
            self._subrun_entry.setText(str(subruns[0]))
            self._subrun_entry.setDisabled(True)
        else:
            tooltip_text = 'Available subruns: '
            tooltip_text += ', '.join(map(str, subruns))
            self._subrun_entry.setToolTip(tooltip_text)

        if len(events) == 1:
            self._event_entry.setText(str(events[0]))
            self._event_entry.setDisabled(True)
        else:
            tooltip_text = 'Available events: '
            tooltip_text += ', '.join(map(str, events))
            self._event_entry.setToolTip(tooltip_text)

    def _on_restore_action(self):
        ''' for now basically just restores closed dock widgets '''
        self.activate()
