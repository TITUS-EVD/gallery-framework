import sys
import datetime

from PyQt5 import Qt, QtGui, QtCore, QtWidgets


'''
# experimental highlight effect when hovering over the menu items
# Not quite working & a bit hacky
class HighlightEffect(QtWidgets.QGraphicsEffect):
    def __init__(self):
        super().__init__()

    def draw(self, painter):
        pixmap = QtGui.QPixmap()

        if self.sourceIsPixmap():
            pixmap, offset = self.sourcePixmap(QtCore.Qt.LogicalCoordinates)
        else:
            pixmap, offset = self.sourcePixmap(QtCore.Qt.DeviceCoordinates)
            painter.setWorldTransform(QtGui.QTransform())
        
        painter.setBrush(QtGui.QColor(0, 0, 0, 255))
        painter.drawRect(pixmap.rect())
        painter.setOpacity(0.5)
        painter.drawPixmap(offset, pixmap)


class HighlightLabel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._widget = None
        self._effect = HighlightEffect

    def set_widget(self, w):
        self._widget = w

    def enterEvent(self, event):
        self._widget.setGraphicsEffect(self._effect())

    def leaveEvent(self, event):
        self._widget.setGraphicsEffect(None)
'''


_SET_FIVEPM_REMINDER = 'General/5 PM reminder'
_SET_SCREENSHOT_SCALE = 'General/Screenshot scale'
_SET_SCREENSHOT_MODE = 'General/Screenshot mode'

class Gui(QtWidgets.QMainWindow):
    def __init__(self, gallery_interface):
        super().__init__()
        self._gi = gallery_interface

        # settings that appear in the settings menu. Modules may add to these
        # via add_setting method

        # On linux, this is equivalent to ~/.config/TITUS/TITUS.conf
        QtCore.QCoreApplication.setApplicationName("TITUS");
        QtCore.QCoreApplication.setOrganizationName("TITUS");

        self._settings = QtCore.QSettings()

        self._settings_layout = QtWidgets.QGridLayout()
        self._screenshot_mode_clip = QtWidgets.QRadioButton('Clipboard')
        self._screenshot_mode_clip.toggled.connect(
            lambda x: self._settings.setValue(_SET_SCREENSHOT_MODE, "Clipboard") if x else 0
        )
        # self._screenshot_mode_clip.toggle()
        self._screenshot_mode_file = QtWidgets.QRadioButton('File')
        self._screenshot_mode_file.toggled.connect(
            lambda x: self._settings.setValue(_SET_SCREENSHOT_MODE, "File") if x else 0
        )
        label = QtWidgets.QLabel(_SET_SCREENSHOT_MODE.split('/')[1])
        self._settings_layout.addWidget(label, 0, 0, 1, 1)
        self._settings_layout.addWidget(self._screenshot_mode_clip, 0, 1, 1, 1)
        self._settings_layout.addWidget(self._screenshot_mode_file, 0, 2, 1, 1)

        label2 = QtWidgets.QLabel(_SET_SCREENSHOT_SCALE.split('/')[1])
        self._screenshot_scale_spin = QtWidgets.QDoubleSpinBox()
        self._screenshot_scale_spin.setRange(1.0, 4.0)
        self._screenshot_scale_spin.setDecimals(1)
        self._screenshot_scale_spin.setSingleStep(0.5)
        self._screenshot_scale_spin.valueChanged.connect(
            lambda x: self._settings.setValue(_SET_SCREENSHOT_SCALE, x)
        )
        self._settings_layout.addWidget(label2, 1, 0, 1, 1)
        self._settings_layout.addWidget(self._screenshot_scale_spin, 1, 1, 1, -1)

        label3 = QtWidgets.QLabel(_SET_FIVEPM_REMINDER.split('/')[1])
        self._fivepm_reminder = QtWidgets.QCheckBox()
        self._fivepm_reminder.stateChanged.connect(
            lambda x: self._settings.setValue(_SET_FIVEPM_REMINDER, x)
        )
        # self._fivepm_reminder.toggle()
        self._settings_layout.addWidget(label3, 2, 0, 1, 1)
        self._settings_layout.addWidget(self._fivepm_reminder, 2, 1, 1, -1)
        self._settings_layout.setRowStretch(self._settings_layout.rowCount(), 1)


        self._restore_from_settings()
        self._settings_dialog = ModuleSettingsDialog(self, self._settings)
        self._settings_dialog.add_settings_layout('General', self._settings_layout)

        self._modules = {}

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.fix_a_drink)
        seconds_to_17 = (17 - datetime.datetime.now().hour - 1) * 60 * 60\
                      + (60 - datetime.datetime.now().minute) * 60\
                      + (60 - datetime.datetime.now().second)
        if seconds_to_17 > 0:
            self._timer.start(seconds_to_17 * 1e3)

        self._central_widget = QtWidgets.QStackedWidget()
        self._central_widget.setMinimumSize(320, 240)
        # scroll_area = QtWidgets.QScrollArea()
        # scroll_area.setMinimumSize(320, 240)

        # modules may access this object thru this.centralWidget()
        self.setCentralWidget(self._central_widget)

        # modules may access this thru this.statusBar()
        self.setStatusBar(QtWidgets.QStatusBar())

        # modules may access this thru this.menuBar()
        self.setMenuBar(QtWidgets.QMenuBar())
        self.menuBar().setNativeMenuBar(True)
        
        self.file_menu = QtWidgets.QMenu("&File", self)
        self.menuBar().addMenu(self.file_menu)

        open_action = QtWidgets.QAction('&Open', self.file_menu)
        open_action.triggered.connect(self._on_open_action)
        self.file_menu.addAction(open_action)

        # screenshot controls
        self.file_menu.addSeparator()
        capture_action = QtWidgets.QAction('Screenshot (view)', self.file_menu)
        # capture_action = QtWidgets.QWidgetAction(self.file_menu)
        # capture_action.setText('Screenshot (view)')
        # label = HighlightLabel("Screenshot (view)")
        # label.set_widget(self._central_widget)
        # capture_action.setDefaultWidget(label)
        capture_action.triggered.connect(self._on_capture_action)
        self.file_menu.addAction(capture_action)

        capture_screen_action = QtWidgets.QAction('Screenshot (window)', self.file_menu)
        capture_screen_action.triggered.connect(self._on_capture_screen_action)
        self.file_menu.addAction(capture_screen_action)

        # exit
        self.file_menu.addSeparator()
        quit_action = QtWidgets.QAction('&Exit', self.file_menu)
        quit_action.triggered.connect(self.closeEvent)
        self.file_menu.addAction(quit_action)

        # edit menu
        self.edit_menu = QtWidgets.QMenu("&Edit", self)
        self.menuBar().addMenu(self.edit_menu)
        pref_action = QtWidgets.QAction('&Preferences', self.edit_menu)
        pref_action.triggered.connect(self._on_preferences_action)
        self.edit_menu.addAction(pref_action)

        self.setWindowTitle('TITUS Event Display')    
        self.setFocus()
        self.show()

        # left and right dock areas should take up the full vertical space
        self.setCorner(QtCore.Qt.TopLeftCorner, QtCore.Qt.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.TopRightCorner, QtCore.Qt.RightDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea)

    @property
    def gallery_interface(self):
        return self._gi

    def fix_a_drink(self):
        self._timer.stop()
        if not self._settings.value(_SET_FIVEPM_REMINDER):
            # maybe next time...
            return

        choice = QtWidgets.QMessageBox.question(self, 'It is 5 pm!',
                                            "Time to fix yourself a drink!",
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            print("Awesome.")
            sys.exit()
        else:
            print("Boring.")

    def closeEvent(self, event):
        self._settings.beginGroup("MainWindow");
        # self._settings.setValue("geometry", self.saveGeometry());
        self._settings.setValue("state", self.saveState(1.0));
        self._settings.endGroup();

        QtCore.QCoreApplication.instance().quit()

    def add_module(self, module):
        module_type = type(module).__name__
        if module_type in self._modules:
            raise ValueError('attempt to add multiple modules of the same class with add_module')
        

        module.connect_gui(self)
        module.initialize()

        self._settings.beginGroup(module.name)
        for key, val in module.settings:
            setting = self._settings.value(key)
            if not setting.isValid():
                self._settings.setValue(key, val)
            else:
                # user must implement this per module
                module.load_setting(setting)
        self._settings.endGroup()
        
        self._modules[module_type] = module

    def _on_open_action(self):
        ''' Show file browser &allow user to open a new file '''
        file_path = str(QtWidgets.QFileDialog.getOpenFileName(filter="ROOT files (*.root)")[0])
        if file_path == '':
            return
        self._gi.set_input_file(file_path)

    def _on_capture_action(self):
        ''' Capture just the central widget '''
        scale = float(self._settings.value(_SET_SCREENSHOT_SCALE))

        w = self._central_widget.width()
        h = self._central_widget.height()
        img = QtGui.QPixmap(w * scale, h * scale);
        img.setDevicePixelRatio(scale);
        self._central_widget.render(img);

        # img = self._central_widget.grab(self._central_widget.rect())
        if self._settings.value(_SET_SCREENSHOT_MODE) == "Clipboard":
            QtWidgets.QApplication.clipboard().setPixmap(img)
            self.statusBar().showMessage('Screenshot copied to clipboard', 3000)
        else:
            fname = self._screenshot_filename()
            file = QtCore.QFile(fname)
            file.open(QtCore.QIODevice.WriteOnly)
            img.save(file, "PNG")
            self.statusBar().showMessage(f'Screenshot saved to {fname}', 3000)

    def _on_capture_screen_action(self):
        ''' Capture the whole application '''
        img = self.grab(self.rect())
        if self._settings.value(_SET_SCREENSHOT_MODE) == "Clipboard":
            QtWidgets.QApplication.clipboard().setPixmap(img)
            self.statusBar().showMessage('Screenshot copied to clipboard', 3000)
        else:
            fname = self._screenshot_filename()
            file = QtCore.QFile(fname)
            file.open(QtCore.QIODevice.WriteOnly);
            img.save(file, "PNG")
            self.statusBar().showMessage(f'Screenshot saved to {fname}', 3000)

    def _on_preferences_action(self):
        self._settings_dialog.exec()

    def _restore_from_settings(self):
        # Attempt to get user's last window settings
        self._settings.beginGroup("MainWindow");
        win_geom = self._settings.value("geometry", QtCore.QByteArray())
        win_state = self._settings.value("state", QtCore.QByteArray())
        if win_geom.isEmpty():
            self.setGeometry(200, 200, 1366, 768);
        else:
            self.restoreGeometry(win_geom)

        if not win_state.isEmpty():
            self.restoreState(win_state, 1)

        self._settings.endGroup();

        # other settings
        if (x := self._settings.value(_SET_SCREENSHOT_MODE)) == '':
            self._screenshot_mode_clip.toggle()
        else:
            if x == 'Clipboard':
                self._screenshot_mode_clip.setChecked(True)
                self._screenshot_mode_file.setChecked(False)
            else:
                self._screenshot_mode_clip.setChecked(False)
                self._screenshot_mode_file.setChecked(True)

        if (x := self._settings.value(_SET_SCREENSHOT_SCALE)) == '':
            self._screenshot_scale_spin.setValue(2.0)
        else:
            self._screenshot_scale_spin.setValue(float(x))

        if (x := self._settings.value(_SET_FIVEPM_REMINDER)) == '':
            self._fivepm_reminder.setChecked(True)
        else:
            self._fivepm_reminder.setChecked(int(x))

    def _screenshot_filename(self):
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"titus_run{self._gi.run()}_sub{self._gi.subrun()}_evt{self._gi.event()}_{now_str}.png"


class ModuleSettingsDialog(QtWidgets.QDialog):
    # saved settings which are not user-configurable
    SKIP_SETTINGS = ['MainWindow']

    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.setMinimumSize(320, 200)
        self._settings = settings
        self._group_layouts = {}

        qbtn = QtWidgets.QDialogButtonBox.Ok
        self.button_box = QtWidgets.QDialogButtonBox(qbtn)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self._widget = QtWidgets.QTabWidget()
        self._layout = QtWidgets.QVBoxLayout()
        self._layout.addWidget(self._widget)

        self.setLayout(self._layout)

    def add_settings_layout(self, group_name, layout):
        self._group_layouts[group_name] = layout
    
    def exec(self):
        self._setup()
        super().exec()

    def _setup(self):
        while self._widget.count():
            w = self._widget.widget(self._widget.currentIndex())
            self._widget.removeTab(self._widget.currentIndex())
            del w

        child_groups = self._settings.childGroups();
        for group in child_groups:
            if group in ModuleSettingsDialog.SKIP_SETTINGS:
                continue

            if group in self._group_layouts.keys():
                widget = QtWidgets.QWidget()
                widget.setLayout(self._group_layouts[group])
                self._widget.addTab(widget, group)

        self._layout.addWidget(self.button_box)

