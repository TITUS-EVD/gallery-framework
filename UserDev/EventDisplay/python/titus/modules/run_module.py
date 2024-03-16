#!/usr/bin/env python3

"""
This module adds run controls for gallery events: forward, back, goto, etc.
Also supports auto-updating from a folder
"""
import os, time
import glob

from PyQt5 import QtWidgets, QtGui, QtCore

from .module import Module


class RunModule(Module):
    def __init__(self):
        super().__init__()

        # if the user wants to auto-update the run and event
        self._min_event_update_time = 20.0
        self._min_file_update_time = 60.0

        # event timer actually increments the events, while UI timer 
        # provides quicker updates to the GUI
        self._event_timer = QtCore.QTimer()
        self._ui_timer = QtCore.QTimer()
        self._ui_timer.timeout.connect(self._auto_advance_timeout)
        self._ui_timer.setInterval(1000)

    def _initialize(self):
        """ Add run controls to the GUI in a DockWidget """
        self.init_auto_advance()

        self._event_dock =  QtWidgets.QDockWidget('Event Controls', self._gui, objectName='_run_dock_event')
        self._event_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._event_dock.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        frame = QtWidgets.QWidget(self._event_dock)
        main_layout = QtWidgets.QVBoxLayout()
        frame.setLayout(main_layout)
        self._event_dock.setWidget(frame)

        # this connection allows dock widget to be restored with activate function
        self._dock_widgets.add(self._event_dock)

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

        # Auto-advance
        self._auto_advance_checkbox = QtWidgets.QCheckBox('Auto-advance')
        self._auto_advance_checkbox.stateChanged.connect(self.toggle_auto_advance)
        main_layout.addWidget(self._auto_advance_checkbox)

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

    def init_auto_advance(self):
        self._event_timer.timeout.connect(self._gi.next)
        self._event_timer.setInterval(self._min_event_update_time * 1000.)

        self._file_handler = FileMonitor(filedir='.',
                                         search_pattern='',
                                         gallery_interface=self._gi,
                                         delay=self._min_file_update_time)
        self._auto_advance_label = QtWidgets.QLabel()

    def toggle_auto_advance(self, checkstate):
        if not checkstate:
            self._event_timer.stop()
            self._ui_timer.stop()
            self._gui.statusBar().removeWidget(self._auto_advance_label)
            self._auto_advance_label.hide()
        else:
            self._event_timer.start()
            self._ui_timer.start()
            self._gui.statusBar().insertPermanentWidget(0, self._auto_advance_label)
            self._auto_advance_label.show()

    def _auto_advance_timeout(self):
        remaining_sec = self._event_timer.remainingTime() / 1000.

        # grab internal event counters from gallery interface
        evts = self._gi._n_entries
        this_evt = self._gi._event + 1
        if this_evt == evts:
            self._auto_advance_checkbox.setCheckState(False)
        self._auto_advance_label.setText(f'Event: {this_evt}/{evts} Next event: {remaining_sec:.0f} s')

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


        '''
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.eventTimeout)
        self._minEventUpdateTime = 30.0 # Seconds
        self._minFileUpdateTime = 60 # Seconds

        self._file_handler = FileHandler(filedir=file_dir,
                                         search_pattern=search_pattern,
                                         ev_manager=self._event_manager,
                                         delay=self._minFileUpdateTime)

        self._stage = None
        self._event_manager.fileChanged.connect(self.drawableProductsChanged)
        self._event_manager.eventChanged.connect(self.update_event_labels)

        self._app = app

        # File Updater
        self._autoFileLabel = QtGui.QLabel("File Update OFF")
        self._autoFileLabel.setStyleSheet("color: red;")
        self._fileUpdateDelayLabel = QtGui.QLabel("Delay (m):")
        self._fileUpdateDelayEntry = QtGui.QLineEdit(str(self._file_handler.get_delay() / 60))
        self._fileUpdateDelayEntry.returnPressed.connect(self.fileUpdateEntryHandler)
        self._fileUpdateDelayEntry.setMaximumWidth(45)
        self._fileUpdatePauseButton = QtGui.QPushButton("START")
        self._fileUpdatePauseButton.clicked.connect(self.fileUpdateButtonHandler)

        # Event Updater
        self._autoRunLabel = QtGui.QLabel("Event Update OFF")
        self._autoRunLabel.setStyleSheet("color: red;")
        self._eventUpdateDelayLabel = QtGui.QLabel("Delay (s):")
        self._eventUpdateDelayEntry = QtGui.QLineEdit("45")
        self._eventUpdateDelayEntry.returnPressed.connect(self.eventUpdateEntryHandler)
        self._eventUpdateDelayEntry.setMaximumWidth(45)
        self._eventUpdatePauseButton = QtGui.QPushButton("START")
        self._eventUpdatePauseButton.clicked.connect(self.eventUpdateButtonHandler)
        '''



class FileMonitor:
    ''' Looks for new files for the live event display '''
    def __init__(self, filedir, search_pattern, gallery_interface, delay=180, do_check=False, hours_alert=1):
        
        self._filedir = filedir
        self._search_pattern = search_pattern
        self._gi = gallery_interface
        # self._message_bar = None
        self._delay = delay
        self._do_check = do_check
        self._hours_alert = hours_alert

        self._first_file = True
        self._current_file = ''

        self._timer = QtCore.QTimer()
        self._timer.setInterval(self._delay * 1000)
        self._timer.timeout.connect(self._callback)

        self._callback()

        if self._do_check:
            self._start_timer()

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, delay):
        ''' Sets the delay to check for new files (in seconds) '''
        self._delay = delay
        self._timer.setInterval(self._delay * 1000)


    def toggle_check(self):
        self._do_check = not self._do_check

        if self._do_check:
            self._start_timer()
            if self._first_time:
                self._callback()
                self._first_time = False

        return self._do_check

    def _callback(self):
        files = self._get_files()

        if not len(files):
            print(f'No files available in {self._filedir}!')
            return

        if files[-1] == self._current_file:
            print('No new file to draw.')
            return

        self._current_file = files[-1]

        print("Switching to file ", self._current_file)
        self._event_manager.setInputFile(self._current_file)

        self._check_time(self._current_file)

    def _get_files(self):
        '''
        Gets all the files in dir _filedir in order of creation (latest last)
        '''
        files = list(filter(os.path.isfile, glob.glob(self._filedir + '/' + self._search_pattern)))
        files.sort(key=lambda x: os.path.getmtime(x))
        return files

    def _check_time(self, file):
        '''
        Checks how old is the last file, and if too old prints a message
        '''

        if self._message_bar is None:
            return

        file_time = os.path.getmtime(file)
        now = time.time()

        hours_old = (now - file_time) / 3600

        # if hours_old > self._hours_alert:
        #     self._message_bar.showMessage(f'The last file appears to be more than {hours_old:0.1f} hour(s) old.')


    def _start_timer(self):
        if self._timer.isActive():
            self._timer.stop()
        self._timer.start()

    def _stop_timer(self):
        if self._timer.isActive():
            self._timer.stop()
