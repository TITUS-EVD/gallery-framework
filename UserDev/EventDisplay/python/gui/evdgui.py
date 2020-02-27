from gui import gui
from pyqtgraph.Qt import QtGui, QtCore
from .boxes import *


class evdgui(gui):

    """
    Inherit the basic gui to extend it
    override the gui to give the display special features:
    """

    def __init__(self, geometry, manager=None, app=None, live=False):
        super(evdgui, self).__init__(geometry)
        if manager is None:
            manager = evd_manager(geometry)
        super(evdgui, self).initManager(manager)
        self._live = live
        self.initUI()
        self._stage = None
        self._event_manager.fileChanged.connect(self.drawableProductsChanged)
        self._event_manager.eventChanged.connect(self.update)
        # self._event_manager.truthLabelChanged.connect(self.updateMessageBar)

        self._app = app

    # override the initUI function to change things:
    def initUI(self):
        super(evdgui, self).initUI()
        # Change the name of the labels for lariat:
        self.update()

    # This function sets up the eastern widget
    def getEastLayout(self):
        # This function just makes a dummy eastern layout to use.
        self._title1 = QtGui.QLabel("TITUS")
        self._title1a = QtGui.QLabel("The event display")
        self._title1b = QtGui.QLabel("for SBN @ Fermilab")
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

        # Draw Wires:
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

        # # Set a box for mcTruth Info
        # self._truthDrawBox = QtGui.QCheckBox("MC Truth")
        # self._truthDrawBox.stateChanged.connect(self.truthDrawBoxWorker)
        # self._eastLayout.addWidget(self._truthDrawBox)


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

        self._eastWidget.setLayout(self._eastLayout)
        self._eastWidget.setMaximumWidth(190)
        self._eastWidget.setMinimumWidth(140)
        return self._eastWidget

    def drawableProductsChanged(self):
        # self.removeItem(self._eastLayout)
        self._eastWidget.close()
        east = self.getEastLayout()
        self.slave.addWidget(east)
        self.update()

        # self._eastLayout.setVisible(False)
        # self._eastLayout.setVisible(True)

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
        # print str

    def noiseFilterWorker(self):
        if self._noiseFilterBox.isChecked():
            self._event_manager.toggleNoiseFilter(True)
        else:
            self._event_manager.toggleNoiseFilter(False)

        self._view_manager.drawPlanes(self._event_manager)

    # def truthDrawBoxWorker(self):
    #     if self._truthDrawBox.isChecked():
    #         self._event_manager.toggleTruth(True)
    #     else:
    #         self._event_manager.toggleTruth(False)

    #     self._event_manager.drawFresh()
    #     # gui.py defines the message bar and handler, connect it to this:

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

