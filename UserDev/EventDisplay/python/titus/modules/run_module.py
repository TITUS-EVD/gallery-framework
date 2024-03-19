#!/usr/bin/env python3

"""
This module adds run controls for gallery events: forward, back, goto, etc.
Also supports auto-updating from a folder
"""
import os, time
import glob

from ROOT import TFile

from PyQt5 import QtWidgets, QtGui, QtCore

from titus.modules import Module
from titus.gui.widgets import ElidedLabel


_SET_AUTOADVANCE_INTERVAL = 'Run/Auto-advance interval'
_SET_AUTOADVANCE_MODE = 'Run/Auto-advance mode'
_SET_FILE_CHECK_INTERVAL = 'Run/File check interval'
_SET_NEXT_FILE_MODE = 'Run/Next file mode'

class RunModule(Module):
    def __init__(self):
        super().__init__()
        self._name = 'Run'

        # event timer waits to increment the events, while UI timer 
        # provides fixed 1s updates to the GUI
        self._event_timer = QtCore.QTimer()
        self._ui_timer = QtCore.QTimer()
        self._ui_timer.timeout.connect(self._ui_timeout)
        self._ui_timer.setInterval(100)

        self._settings_defaults = {
            _SET_AUTOADVANCE_MODE: 'Interval',
            _SET_AUTOADVANCE_INTERVAL: 20,
            _SET_FILE_CHECK_INTERVAL: 30,
            _SET_NEXT_FILE_MODE: 'Sequential',
        }
        self._init_settings_page()

    def _init_settings_page(self):
        self._settings_layout = QtWidgets.QGridLayout()
        self._autoadvance_mode_interval = QtWidgets.QRadioButton('Interval')
        self._autoadvance_mode_interval.toggled.connect(
            lambda x: self._settings.setValue(_SET_AUTOADVANCE_MODE, 'Interval') if x else 0
        )
        self._autoadvance_mode_timestamp = QtWidgets.QRadioButton('Timestamp')
        self._autoadvance_mode_timestamp.toggled.connect(
            lambda x: self._settings.setValue(_SET_AUTOADVANCE_MODE, 'Timestamp') if x else 0
        )
        self._autoadvance_mode_timestamp.setEnabled(False)

        self._btngrp = QtWidgets.QButtonGroup()
        self._btngrp.addButton(self._autoadvance_mode_interval)
        self._btngrp.addButton(self._autoadvance_mode_timestamp)

        label = QtWidgets.QLabel(_SET_AUTOADVANCE_MODE.split('/')[1])
        self._settings_layout.addWidget(label, 0, 0, 1, 1)
        self._settings_layout.addWidget(self._autoadvance_mode_interval, 0, 1, 1, 1)
        self._settings_layout.addWidget(self._autoadvance_mode_timestamp, 0, 2, 1, 1)

        label2 = QtWidgets.QLabel(f'{_SET_AUTOADVANCE_INTERVAL.split("/")[1]} (s)')
        self._autoadvance_interval = QtWidgets.QSpinBox()
        self._autoadvance_interval.setRange(5, 600)
        self._autoadvance_interval.setSingleStep(1)
        self._autoadvance_interval.setValue(self._settings_defaults[_SET_AUTOADVANCE_INTERVAL])
        self._autoadvance_interval.valueChanged.connect(
            lambda x: self._settings.setValue(_SET_AUTOADVANCE_INTERVAL, x)
        )

        self._settings_layout.addWidget(label2, 1, 0, 1, 1)
        self._settings_layout.addWidget(self._autoadvance_interval, 1, 1, 1, -1)

        label4 = QtWidgets.QLabel(_SET_NEXT_FILE_MODE.split('/')[1])
        self._nextfile_mode_sequential = QtWidgets.QRadioButton('Sequential')
        self._nextfile_mode_sequential.toggled.connect(
            lambda x: self._settings.setValue(_SET_NEXT_FILE_MODE, 'Sequential') if x else 0
        )
        self._nextfile_mode_newest = QtWidgets.QRadioButton('Newest')
        self._nextfile_mode_newest.toggled.connect(
            lambda x: self._settings.setValue(_SET_NEXT_FILE_MODE, 'Newest') if x else 0
        )

        self._btngrp2 = QtWidgets.QButtonGroup()
        self._btngrp2.addButton(self._nextfile_mode_sequential)
        self._btngrp2.addButton(self._nextfile_mode_newest)

        self._settings_layout.addWidget(label4, 2, 0, 1, 1)
        self._settings_layout.addWidget(self._nextfile_mode_sequential, 2, 1, 1, 1)
        self._settings_layout.addWidget(self._nextfile_mode_newest, 2, 2, 1, 1)

        label3 = QtWidgets.QLabel(f'{_SET_FILE_CHECK_INTERVAL.split("/")[1]} (s)')
        self._new_file_check_interval = QtWidgets.QSpinBox()
        self._new_file_check_interval.setRange(10, 600)
        self._new_file_check_interval.setSingleStep(1)
        self._new_file_check_interval.setValue(self._settings_defaults[_SET_FILE_CHECK_INTERVAL])
        self._new_file_check_interval.valueChanged.connect(
            lambda x: self._settings.setValue(_SET_FILE_CHECK_INTERVAL, x)
        )

        self._settings_layout.addWidget(label3, 3, 0, 1, 1)
        self._settings_layout.addWidget(self._new_file_check_interval, 3, 1, 1, -1)

        self._settings_layout.setRowStretch(self._settings_layout.rowCount(), 1)

    def _initialize(self):
        """ Add run controls to the GUI in a DockWidget """
        # modules update on file change automatically but must explicitly set
        # update on directory change if desired
        self._gi.dirChanged.connect(self.update)

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

        # file/directory labels
        self._file_label = ElidedLabel('File: None')
        self._dir_label = ElidedLabel('Directory: None')
        main_layout.addWidget(self._file_label)
        main_layout.addWidget(self._dir_label)

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

    def restore_from_settings(self):
        x = self._settings.value(_SET_AUTOADVANCE_INTERVAL,
                                 self._settings_defaults[_SET_AUTOADVANCE_INTERVAL])
        self._autoadvance_interval.setValue(int(x))

        mode = self._settings.value(_SET_AUTOADVANCE_MODE,
                                 self._settings_defaults[_SET_AUTOADVANCE_MODE])
        if mode == 'Interval':
            self._autoadvance_mode_interval.toggle()
        else:
            self._autoadvance_mode_timestamp.toggle()

        x = self._settings.value(_SET_FILE_CHECK_INTERVAL,
                                 self._settings_defaults[_SET_FILE_CHECK_INTERVAL])
        self._new_file_check_interval.setValue(int(x))

        mode2 = self._settings.value(_SET_NEXT_FILE_MODE,
                                 self._settings_defaults[_SET_NEXT_FILE_MODE])
        if mode2 == 'Sequential':
            self._nextfile_mode_sequential.toggle()
        else:
            self._nextfile_mode_newest.toggle()

    def init_auto_advance(self):
        self._event_timer.timeout.connect(self._auto_advance_timeout)
        self._file_handler = FileMonitor(filedir=None, search_pattern='*.root',
                                         gallery_interface=self._gi, delay=self._settings_defaults[_SET_FILE_CHECK_INTERVAL])
        self._auto_advance_label = QtWidgets.QLabel()

    def toggle_auto_advance(self, checkstate):
        if not checkstate:
            self._event_timer.stop()
            self._ui_timer.stop()
            self._gui.statusBar().removeWidget(self._auto_advance_label)
            self._auto_advance_label.hide()
            self._file_handler.stop()
            return
        
        interval = int(self._settings.value(_SET_AUTOADVANCE_INTERVAL,
                                            self._settings_defaults[_SET_AUTOADVANCE_INTERVAL]))
        self._event_timer.setInterval(interval * 1000)
        file_interval = int(self._settings.value(_SET_FILE_CHECK_INTERVAL,
                                                 self._settings_defaults[_SET_FILE_CHECK_INTERVAL]))
        self._file_handler.delay = file_interval
        self._file_handler.filedir = self._gi.current_directory
        self._file_handler.mode = self._settings.value(_SET_NEXT_FILE_MODE, self._settings_defaults[_SET_NEXT_FILE_MODE])
        self._file_handler.start()

        self._event_timer.start()
        self._ui_timer.start()
        self._auto_advance_label.setText('')
        self._gui.statusBar().insertPermanentWidget(0, self._auto_advance_label)
        self._auto_advance_label.show()

    def _auto_advance_timeout(self):
        # grab internal event counters from gallery interface
        evts = self._gi._n_entries
        this_evt = self._gi._entry + 1
        if this_evt < evts:
            self._gi.next()
        else:
            # ask the file handler if there's a new file
            self._file_handler.callback()
            evts = self._gi._n_entries
            this_evt = self._gi._entry

    def _ui_timeout(self):
        evts = self._gi._n_entries
        this_evt = self._gi._entry + 1
        remaining_sec = self._event_timer.remainingTime() / 1000.
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

        # disable next/back buttons to indicate end or beginning of file
        evts = self._gi._n_entries
        this_evt = self._gi._entry + 1
        self._next_button.setEnabled(this_evt < evts)
        self._prev_button.setEnabled(this_evt > 1)

        self._larlite_event_entry.setText(str(self._gi.event()))

        self._event_label.setText(f'Event: {self._gi.event()}')
        self._run_label.setText(f'Run: {self._gi.run()}')
        self._subrun_label.setText(f'Subrun: {self._gi.subrun()}')

        self._file_label.setText(f'File: {os.path.basename(self._gi.current_file)}')

        # directory: Display current absolute, otherwise shorter of 
        # relpath and absolute path
        dir_path = os.getcwd()
        if (x := os.path.relpath(self._gi.current_directory)) != '.':
            if len(x) > len(self._gi.current_directory):
                dir_path = self._gi.current_directory
            else:
                dir_path = x
        self._dir_label.setText(f'Scan dir.: {dir_path}')

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


class FileMonitor:
    ''' Looks for new files for the live event display '''

    _STATUS_NONE = ''
    _STATUS_WAITING = 'Waiting for new files'

    def __init__(self, filedir, search_pattern, gallery_interface, delay=180, do_check=False, hours_alert=1):
        self._filedir = filedir
        self._search_pattern = search_pattern
        self._gi = gallery_interface
        # self._message_bar = None
        self._delay = delay
        self._do_check = do_check
        self._hours_alert = hours_alert
        self._status = FileMonitor._STATUS_NONE

        # file opening mode: either "Sequential" to move to the next file
        # in order, or "Newest" to always skip to the newest file
        self._mode = 'Sequential'

        self._timer = QtCore.QTimer()
        self._timer.setInterval(self._delay * 1000)
        self._timer.timeout.connect(self._get_files)

        if self._do_check:
            self._start_timer()
        self._files = []

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, delay):
        ''' Sets the delay to check for new files (in seconds) '''
        self._delay = delay
        self._timer.setInterval(self._delay * 1000)

    @property
    def filedir(self):
        return self._filedir

    @filedir.setter
    def filedir(self, val):
        self._filedir = val

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        if val not in ['Sequential', 'Newest']:
            raise ValueError(f'Unsupported file monitor mode {val}')
        self._mode = val

    def start(self):
        self._start_timer()
        self._get_files()

    def stop(self):
        self._stop_timer()

    def callback(self):
        # only go to next file if we are on the last event of the current one
        evts = self._gi._n_entries
        this_evt = self._gi._entry + 1
        if this_evt < evts:
            return

        if self._filedir is None:
            self._filedir = self._gi.current_directory

        if not self._files:
            print(f'No files available in {self._filedir}!')
            return

        next_file = None
        if self._mode == 'Newest':
            if self._files[-1][0] == self._gi.current_file:
                self._status = FileMonitor._STATUS_WAITING
                print(self._status)
                return

            next_file = self._files[-1][0]
        else:
            # find the next file after the current one
            current_file_time = os.path.getmtime(self._gi.current_file)
            ft = self._files[0][1]
            idx = 0
            while current_file_time >= ft and idx < len(self._files):
                ft = self._files[idx][1]
                idx += 1

            if current_file_time >= ft:
                self._status = FileMonitor._STATUS_WAITING
                print(self._status)
                return
            next_file = self._files[idx - 1][0]

        if next_file is None:
            print('Warning: Next file was not set')
            return

        self.clear_status()

        print("Switching to file ", next_file)
        self._gi.set_input_file(next_file)
        self._check_time(next_file)

    def _get_files(self):
        '''
        Gets all the files in dir _filedir in order of creation (latest last)
        '''

        current_file_time = os.path.getmtime(self._gi.current_file)

        files = list(filter(os.path.isfile, glob.glob(self._filedir + '/' + self._search_pattern)))
        self._files = []
        for f in files:
            if os.path.getmtime(f) < current_file_time:
                continue

            # make sure this is an artroot file
            # TODO essentially copies the check from ping_file in
            # gallery_interface. make the method in gallery interface const to
            # avoid the duplication
            try:
                tf = TFile(f)
                e = tf.Get("Events")
                ev_aux_b = e.GetBranch("EventAuxiliary")
            except (OSError, AttributeError):
                print(f"\033[91m WARNING\033[0m Could not open {f}, skipping")
                continue

            self._files.append([f, os.path.getmtime(f)])

    def _check_time(self, file):
        '''
        Checks how old is the last file, and if too old prints a message
        '''

        # if self._message_bar is None:
        #     return

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

    @property
    def status(self):
        return self._status

    def clear_status(self):
        self._status = FileMonitor._STATUS_NONE
