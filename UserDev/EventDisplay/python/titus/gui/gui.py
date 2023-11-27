import datetime

from PyQt5 import Qt, QtGui, QtCore, QtWidgets


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
        quit_action = QtWidgets.QAction('&Exit', self.file_menu)
        quit_action.triggered.connect(self.closeEvent)
        self.file_menu.addAction(quit_action)

        self.resize(1366, 768)
        self.setWindowTitle('TITUS Event Display')    
        self.setFocus()
        self.show()

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
