import os, time
import glob
from pyqtgraph.Qt import QtCore

class FileHandler():
    '''
    Looks for new files for the live event display
    '''

    def __init__(self,
                 filedir,
                 search_pattern,
                 ev_manager,
                 delay=180,
                 do_check=False,
                 hours_alert=1):
        '''

        - filedir: the directory where to look for files
        - search_pattern: the pattern used to reach for files
        - ev_manager: the ene=vent manager
        - delay: the delay for checking new files (seconds)
        - do_check: checks for new files
        - hours_alert: if the last files was created hours_alert before now, trigger an alert
        '''

        self._filedir = filedir
        self._search_pattern = search_pattern
        self._event_manager = ev_manager
        self._message_bar = None
        self._delay = delay
        self._do_check = do_check
        self._hours_alert = hours_alert

        self._first_time = True
        self._current_file = ''

        self._timer = QtCore.QTimer()
        self._timer.setInterval(self._delay * 1000)
        self._timer.timeout.connect(self._callback)

        self._callback()

        if self._do_check:
            self._start_timer()

    def connect_message_bar(self, message_bar):
        '''
        Connects the massage bar.

        Arguments:
        - message_bar: the evd message bar, for alert messages
        '''

        self._message_bar = message_bar


    def set_delay(self, delay):
        '''
        Sets the delay to check for new files (in seconds)
        '''
        print('Setting delay to', delay)
        self._delay = delay
        self._timer.setInterval(self._delay * 1000)

    def get_delay(self):
        '''
        Returns the delay
        '''
        return self._delay

    def change_status(self):
        '''
        Changes the status (check for new files or not)
        Returns:

        - the current status (after change)
        '''
        self._do_check = not self._do_check

        if self._do_check:
            self._start_timer()
            if self._first_time:
                self._callback()
                self._first_time = False

        return self._do_check

    def _callback(self):
        '''
        THe main callback function that does the job
        '''
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

        if hours_old > self._hours_alert:
            self._message_bar.showMessage(f'The last file appears to be more than {hours_old:0.1f} hour(s) old.')


    def _start_timer(self):
        '''
        Starts the timer
        '''

        if self._timer.isActive():
            self._timer.stop()
        self._timer.start()

    def _stop_timer(self):
        '''
        Stops the timer
        '''

        if self._timer.isActive():
            self._timer.stop()
