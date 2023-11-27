from gui import gui
from PyQt5 import QtGui, QtCore, QtWidgets
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
        # Change the name of the labels for this detector
        if self._geometry:
            self._detector_label.setText(\
                f'Detector: {self._geometry.name().upper()}')
        self.update()

        # TODO Move this info to about page
        # This function just makes a dummy eastern layout to use.
        # self._title1 = QtWidgets.QLabel("TITUS")
        # self._title1a = QtWidgets.QLabel("The event display")
        # self._title1b = QtWidgets.QLabel("for SBN @ Fermilab")
        # self._title1c = QtWidgets.QLabel("Version " + self.get_git_version())

    def setupDrawControls(self):
        dock =  QtWidgets.QDockWidget('Draw Controls', self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        frame = QtWidgets.QWidget(dock)
        main_layout = QtWidgets.QVBoxLayout()
        frame.setLayout(main_layout)
        dock.setWidget(frame)

        # stage selection, horizontally centered
        stage_layout = QtWidgets.QHBoxLayout()
        stage_layout.addStretch()
        stage_layout.addWidget(QtWidgets.QLabel("Stage:"))
        self._stageSelection = QtWidgets.QComboBox()
        self._stageSelection.activated[str].connect(self.stageSelectHandler)
        # Make sure "all" is default and on top:
        self._stageSelection.addItem("all")
        for stage in self._event_manager.getStages():
            if stage != "all":
                self._stageSelection.addItem(stage)
        stage_layout.addWidget(self._stageSelection)
        stage_layout.addStretch()
        main_layout.addLayout(stage_layout)


        # None/Wire/RawDigit/OpDetWaveform Radio Buttons
        # None & OpDetWaveform have no drop down
        wire_button_group = QtWidgets.QButtonGroup()

        self._none_wire_button = QtWidgets.QRadioButton("None")
        self._none_wire_button.clicked.connect(self.wireChoiceWorker)
        wire_button_group.addButton(self._none_wire_button)

        self._wire_button = QtWidgets.QRadioButton("Wire")
        self._wire_button.clicked.connect(self.wireChoiceWorker)
        wire_button_group.addButton(self._wire_button)

        products = self._event_manager.get_products('recob::Wire')
        default_products = self._event_manager.get_default_products('recob::Wire')
        self._wire_choice = waveformBox(self, 'recob::Wire', products, default_products)

        # Draw Raw Digit
        self._raw_digit_button = QtWidgets.QRadioButton("Raw Digit")
        self._raw_digit_button.clicked.connect(self.wireChoiceWorker)
        wire_button_group.addButton(self._raw_digit_button)
        products = self._event_manager.get_products('raw::RawDigit')
        default_products = self._event_manager.get_default_products('raw::RawDigit')
        self._raw_digit_choice = waveformBox(self, 'raw::RawDigit', products, default_products)
        raw_digit_layout = QtWidgets.QHBoxLayout()

        self._opdetWvfButton = QtWidgets.QRadioButton("OpDetWaveform")
        self._opdetWvfButton.clicked.connect(self.opdetWvfChoiceWorker)
        wire_button_group.addButton(self._opdetWvfButton)

        wire_choice_layout = QtWidgets.QGridLayout()
        wire_choice_layout.addWidget(self._none_wire_button, 0, 0, 1, 1)
        wire_choice_layout.addWidget(self._wire_button, 1, 0, 1, 1)
        wire_choice_layout.addWidget(self._wire_choice, 1, 1, 1, 1)
        self._wire_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        wire_choice_layout.addWidget(self._raw_digit_button, 2, 0, 1, 1)
        wire_choice_layout.addWidget(self._raw_digit_choice, 2, 1, 1, 1)
        wire_choice_layout.addWidget(self._opdetWvfButton, 3, 0, 1, 1)
        self._raw_digit_choice.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        main_layout.addLayout(wire_choice_layout)

        # Set the default to be no wires
        self._none_wire_button.toggle()

        # Microboone only:
        # TODO re-enable this & add to geometry
        # if self._geometry.name() == "uboone":
        #     self._noiseFilterBox = QtWidgets.QCheckBox("Noise Filter")
        #     self._noiseFilterBox.stateChanged.connect(self.noiseFilterWorker)

        # # Set a box for mcTruth Info
        # self._truthDrawBox = QtWidgets.QCheckBox("MC Truth")
        # self._truthDrawBox.stateChanged.connect(self.truthDrawBoxWorker)
        # self._eastLayout.addWidget(self._truthDrawBox)


        # Now we get the list of items that are drawable:
        drawable_layout = QtWidgets.QFormLayout()
        drawable_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        # drawable_frame.setLayout(drawable_layout)

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
            thisBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            drawable_layout.addRow(product, thisBox)

        main_layout.addLayout(drawable_layout)
        main_layout.addStretch()

        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.resizeDocks([dock], [350], QtCore.Qt.Horizontal)

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
        if self._none_wire_button.isChecked():
            self._event_manager.toggleWires(None)
        if self._wire_button.isChecked():
            self._event_manager.toggleWires('wire',stage=self._stage, producers=activeProducers)
        if self._raw_digit_button.isChecked():
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
        print('recoBoxHandler', text)
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

