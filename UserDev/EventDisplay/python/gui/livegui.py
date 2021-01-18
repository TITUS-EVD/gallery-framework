from gui import gui
from pyqtgraph.Qt import QtGui, QtCore
from .boxes import *
from .file_handler import FileHandler

class livegui(gui):

    """
    Inherit the basic gui to extend it
    override the gui to give the display special features.
    Live GUI
    """

    def __init__(self, geometry, manager=None, app=None, file_dir='./', search_pattern='*.root'):
        super(livegui, self).__init__(geometry)
        if manager is None:
            manager = live_evd_manager_2D(geometry)
        super(livegui, self).initManager(manager)

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
        self._event_manager.eventChanged.connect(self.update)

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

        self.initUI()


    def eventTimeout(self):
        self._event_manager.next()

    # override the initUI function to change things:
    def initUI(self):
        super(livegui, self).initUI()

        self._file_handler.connect_message_bar(self._messageBar)

        self.update()

    # This function sets up the eastern widget
    def getEastLayout(self):
        super(livegui, self).getEastLayout()


        self._title1 = QtGui.QLabel("TITUS <i>Live</i>")
        self._title1a = QtGui.QLabel("The event display")
        self._title1b = QtGui.QLabel("for SBN @ Fermilab")
        self._title1c = QtGui.QLabel("Version " + self.get_git_version())
        geoName = self._geometry.name()
        self._title2 = QtGui.QLabel('Detector: '+geoName.upper())
        font = self._title1.font()
        font.setBold(True)
        self._title1.setFont(font)
        self._title2.setFont(font)



        self._eastWidget = QtGui.QWidget()
        # This is the total layout
        self._eastLayout = QtGui.QVBoxLayout()
        # add the information sections:
        self._eastLayout.addWidget(self._title1)
        self._eastLayout.addWidget(self._title1a)
        self._eastLayout.addWidget(self._title1b)
        self._eastLayout.addWidget(self._title1c)
        self._eastLayout.addWidget(self._title2)
        self._eastLayout.addStretch(1)

        self._stageLabel = QtGui.QLabel("Stage:")
        self._stageSelection = QtGui.QComboBox()
        self._stageSelection.activated[str].connect(self.stageSelectHandler)
        # Make sure "all" is default and on top:
        self._stageSelection.addItem("all")
        for stage in self._event_manager.getStages():
            if stage != "all":
                self._stageSelection.addItem(stage)

        self._eastLayout.addWidget(self._stageLabel)
        self._eastLayout.addWidget(self._stageSelection)
        self._eastLayout.addStretch(1)

        # The wires are a special case.
        # Use a check box to control wire drawing
        self._wireButtonGroup = QtGui.QButtonGroup()
        # Draw no wires:
        self._noneWireButton = QtGui.QRadioButton("None")
        self._noneWireButton.clicked.connect(self.wireChoiceWorker)
        self._wireButtonGroup.addButton(self._noneWireButton)

        # Draw Wires:
        self._wireButton = QtGui.QRadioButton("Wire")
        self._wireButton.clicked.connect(self.wireChoiceWorker)
        self._wireButtonGroup.addButton(self._wireButton)
        products = self._event_manager.get_products('recob::Wire')
        default_products = self._event_manager.get_default_products('recob::Wire')
        self._wireChoice = waveformBox(self, 'recob::Wire', products, default_products)
        self._wireLayout = QtGui.QHBoxLayout()
        self._wireLayout.addWidget(self._wireButton)
        self._wireLayout.addWidget(self._wireChoice)

        # Draw Raw Digit
        self._rawDigitButton = QtGui.QRadioButton("Raw Digit")
        self._rawDigitButton.clicked.connect(self.wireChoiceWorker)
        self._wireButtonGroup.addButton(self._rawDigitButton)
        products = self._event_manager.get_products('raw::RawDigit')
        default_products = self._event_manager.get_default_products('raw::RawDigit')
        self._rawDigitChoice = waveformBox(self, 'raw::RawDigit', products, default_products)
        self._rawDigitLayout = QtGui.QHBoxLayout()
        self._rawDigitLayout.addWidget(self._rawDigitButton)
        self._rawDigitLayout.addWidget(self._rawDigitChoice)

        # Draw Optical Waveforms:
        self._opdetWvfButton = QtGui.QRadioButton("OpDetWaveform")
        self._opdetWvfButton.clicked.connect(self.opdetWvfChoiceWorker)
        self._wireButtonGroup.addButton(self._opdetWvfButton)

        # Make a layout for this stuff:
        self._wireChoiceLayout = QtGui.QVBoxLayout()
        self._wireChoiceLabel = QtGui.QLabel("Draw Options")
        self._wireChoiceLayout.addWidget(self._wireChoiceLabel)
        self._wireChoiceLayout.addWidget(self._noneWireButton)
        self._wireChoiceLayout.addLayout(self._wireLayout)
        # self._wireChoiceLayout.addWidget(self._wireButton)
        self._wireChoiceLayout.addLayout(self._rawDigitLayout)
        # self._wireChoiceLayout.addWidget(self._rawDigitButton)
        self._wireChoiceLayout.addWidget(self._opdetWvfButton)

        self._eastLayout.addLayout(self._wireChoiceLayout)

        # Set the default to be no wires
        self._noneWireButton.toggle()

        # Microboone only:
        if self._geometry.name() == "uboone":
            self._noiseFilterBox = QtGui.QCheckBox("Noise Filter")
            self._noiseFilterBox.stateChanged.connect(self.noiseFilterWorker)
            self._eastLayout.addWidget(self._noiseFilterBox)

        # Now we get the list of items that are drawable:
        drawableProducts = self._event_manager.getDrawableProducts()
        # print drawableProducts
        self._listOfRecoBoxes = []
        for product in drawableProducts:
            thisBox = recoBox(self,
                              product,
                              drawableProducts[product][1],
                              self._event_manager.getProducers(
                                  drawableProducts[product][1]))
            self._listOfRecoBoxes.append(thisBox)
            thisBox.activated[str].connect(self.recoBoxHandler)
            self._eastLayout.addWidget(thisBox)
        self._eastLayout.addStretch(2)

        # Add the auto file switch stuff:
        self._eastLayout.addWidget(self._autoFileLabel)
        autoFileLayout = QtGui.QHBoxLayout()
        autoFileLayout.addWidget(self._fileUpdateDelayLabel)
        autoFileLayout.addWidget(self._fileUpdateDelayEntry)
        self._eastLayout.addLayout(autoFileLayout)
        self._eastLayout.addWidget(self._fileUpdatePauseButton)
        self._eastLayout.addStretch(1)

        # Add the auto event switch stuff:
        self._eastLayout.addWidget(self._autoRunLabel)
        autoDelayLayout = QtGui.QHBoxLayout()
        autoDelayLayout.addWidget(self._eventUpdateDelayLabel)
        autoDelayLayout.addWidget(self._eventUpdateDelayEntry)
        self._eastLayout.addLayout(autoDelayLayout)
        self._eastLayout.addWidget(self._eventUpdatePauseButton)
        self._eastLayout.addStretch(1)

        self._eastWidget.setLayout(self._eastLayout)
        self._eastWidget.setMaximumWidth(190)
        self._eastWidget.setMinimumWidth(140)

        return self._eastWidget


    def fileUpdateEntryHandler(self):
        try:
            delay = float(self._fileUpdateDelayEntry.text())
        except Exception as e:
            delay = self._minFileUpdateTime
            self._eventUpdateDelayEntry.setText(str(delay))
            self._file_handler.set_delay(delay * 60)
            return

        if delay < self._minFileUpdateTime / 60:
            delay = self._minFileUpdateTime / 60
            print('Cannot set delay to a value smaller than', self._minFileUpdateTime, ' s.')
            self._fileUpdateDelayEntry.setText(str(delay))
        self._file_handler.set_delay(delay * 60)



    def fileUpdateButtonHandler(self):

        checking = self._file_handler.change_status()

        if checking:
            self._fileUpdatePauseButton.setText("PAUSE")
            self._autoFileLabel.setText("File Update ON")
            self._autoFileLabel.setStyleSheet("color: black;")
        else:
            self._fileUpdatePauseButton.setText("START")
            self._autoFileLabel.setText("File Update OFF")
            self._autoFileLabel.setStyleSheet("color: red;")


    def eventUpdateEntryHandler(self):
        try:
            delay = float(self._eventUpdateDelayEntry.text())
        except Exception as e:
            delay = self._minEventUpdateTime
            self._eventUpdateDelayEntry.setText(str(delay))
            return
        if delay < self._minEventUpdateTime:
            delay = self._minEventUpdateTime
            self._eventUpdateDelayEntry.setText(str(delay))
            return


    def eventUpdateButtonHandler(self):
        if self._timer.isActive():
            self._timer.stop()
            self._eventUpdatePauseButton.setText("START")
            self._autoRunLabel.setText("Event Update OFF")
            self._autoRunLabel.setStyleSheet("color: red;")
        else:
            try:
                delay = float(self._eventUpdateDelayEntry.text())
            except Exception as e:
                delay = self._minEventUpdateTime
            if delay < self._minEventUpdateTime:
                delay = self._minEventUpdateTime
            self._eventUpdateDelayEntry.setText(str(delay))
            self._eventUpdatePauseButton.setText("PAUSE")
            self._timer.setInterval(delay*1000)
            self._timer.start()
            self._autoRunLabel.setText("Event Update ON")
            self._autoRunLabel.setStyleSheet("color: black;")


    def drawableProductsChanged(self):
        # self._eastWidget.close()
        # east = self.getEastLayout()
        # self.slave.addWidget(east)
        self.update()
        self.repaint()


    def wireChoiceWorker(self, status, activeProducers=None):

        self._view_manager.setDrawingRawDigits(False)
        if self._noneWireButton.isChecked():
            self._event_manager.toggleWires(None)
        if self._wireButton.isChecked():
            self._event_manager.toggleWires('wire',stage=self._stage, producers=activeProducers)
        if self._rawDigitButton.isChecked():
            self._event_manager.toggleWires('rawdigit',stage=self._stage, producers=activeProducers)
            self._view_manager.setDrawingRawDigits(True)

        self._view_manager.drawPlanes(self._event_manager)


    def opdetWvfChoiceWorker(self):
        sender = self.sender()
        if sender == self._opdetWvfButton:
            self._event_manager.toggleOpDetWvf('opdetwaveform', stage=self._stage)
        self._view_manager.drawOpDetWvf(self._event_manager)


    def stageSelectHandler(self, _str):
        self._stage = _str
        if _str == "all":
            self._stage = None
        for box in self._listOfRecoBoxes:
            box.selectStage(_str)

    def noiseFilterWorker(self):
        if self._noiseFilterBox.isChecked():
            self._event_manager.toggleNoiseFilter(True)
        else:
            self._event_manager.toggleNoiseFilter(False)

        self._view_manager.drawPlanes(self._event_manager)


    def splitTracksWorker(self):
        if self._spliTracksOption.isChecked():
            self._tracksOnBothTPCs = True
            for box in self._listOfRecoBoxes:
                if box.name() == 'MCTrack' or box.name() == 'Track':
                    box.emitSignal(box.currentProducer())
        else:
            self._tracksOnBothTPCs = False
            for box in self._listOfRecoBoxes:
                if box.name() == 'MCTrack' or box.name() == 'Track':
                    box.emitSignal(box.currentProducer())

    def recoBoxHandler(self, text):
        sender = self.sender()
        # Get the full product obj for this:
        prod = sender.productObj(text, self._stage)

        if text == "--Select--" or text == "--None--" or text == None:
            self._event_manager.redrawProduct(sender.name(), None, self._view_manager)
            self.specialHandles(sender.name(), False)
            return
        else:
            self._event_manager.redrawProduct(sender.name(), prod, self._view_manager)
            self.specialHandles(sender.name(), True)

    def subtractPedestalWorker(self):
        if self._subtractPedestal.isChecked():
            self._subtract_pedestal = True
        else:
            self._subtract_pedestal = False

        if self._view_manager.drawingRawDigits():
            self._event_manager.toggleWires('rawdigit',
                                            stage=self._stage,
                                            subtract_pedestal=self._subtract_pedestal)

            self._view_manager.drawPlanes(self._event_manager)

    def specialHandles(self, name, visibility):
        '''
        Here we handle all cases specific to
        the product we are drawing
        '''
        if name == 'MCTrack':
            if visibility:
                self._spliTracksOption.setVisible(False)
            else:
                self._spliTracksOption.setVisible(False)
        if name == 'Track':
            if visibility:
                self._spliTracksOption.setVisible(False)
            else:
                self._spliTracksOption.setVisible(False)

