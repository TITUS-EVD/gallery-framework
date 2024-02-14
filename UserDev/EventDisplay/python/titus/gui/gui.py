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


class Gui(QtWidgets.QMainWindow):
    def __init__(self, gallery_interface):
        super().__init__()
        self._gi = gallery_interface
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

        self.resize(1366, 768)
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
        choice = QtWidgets.QMessageBox.question(self, 'It is 5 pm!',
                                            "Time to fix yourself a drink!",
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            print("Awesome.")
            sys.exit()
        else:
            print("Boring.")

    def closeEvent(self, event):
        QtCore.QCoreApplication.instance().quit()

    def add_module(self, module):
        module_name = type(module).__name__
        if module_name in self._modules:
            raise ValueError('attempt to add multiple modules of the same class with add_module')

        module.connect_gui(self)
        module.initialize()
        self._modules[module_name] = module

    def _on_open_action(self):
        ''' Show file browser &allow user to open a new file '''
        file_path = str(QtWidgets.QFileDialog.getOpenFileName(filter="ROOT files (*.root)")[0])
        self._gi.set_input_file(file_path)

    def _on_capture_action(self):
        ''' Capture just the central widget '''
        img = self._central_widget.grab(self._central_widget.rect())
        QtWidgets.QApplication.clipboard().setPixmap(img)

    def _on_capture_screen_action(self):
        ''' Capture the whole application '''
        img = self.grab(self.rect())
        QtWidgets.QApplication.clipboard().setPixmap(img)
