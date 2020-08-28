from pyqtgraph.Qt import QtGui, QtCore

class ComboBoxWithKeyConnect(QtGui.QComboBox):

    def __init__(self):
        super(ComboBoxWithKeyConnect, self).__init__()

    def connectOwnerKPE(self, kpe):
        self._owner_KPE = kpe

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Up:
            super(ComboBoxWithKeyConnect, self).keyPressEvent(e)
            return
        if e.key() == QtCore.Qt.Key_Down:
            super(ComboBoxWithKeyConnect, self).keyPressEvent(e)
            return
        else:
            self._owner_KPE(e)

class waveformBox(QtGui.QWidget):
    activated = QtCore.pyqtSignal(str)

    def __init__(self, owner, name, products, default_products=[]):
        super(waveformBox, self).__init__()
        # self._label = QtGui.QLabel()
        self._name = name
        self._owner = owner

        self._box = QtGui.QToolButton(self)
        self._box.setMinimumWidth(55)
        self._box.setText('Select')
        self._toolmenu = QtGui.QMenu(self)
        # self.setMinimumWidth(100)
        self._actions = []
        if products is not None:
            for i, product in enumerate(products):
                action = self._toolmenu.addAction(product.fullName())
                action.setCheckable(True)
                if product in default_products:
                    action.setChecked(True)
                action.triggered.connect(self.emitSignal)
                self._actions.append(action)
        else:
            action = self._toolmenu.addAction("No " + name + " available to draw.")
            action.setCheckable(False)

        self._box.setMenu(self._toolmenu)
        self._box.setPopupMode(QtGui.QToolButton.InstantPopup)

        # This is the widget itself, so set it up
        self._layout = QtGui.QHBoxLayout()
        # self._layout.addWidget(self._label)
        # self._layout.addWidget(self._box)
        self._layout.addWidget(self._box)
        self.setLayout(self._layout)


    def emitSignal(self, text):
        # self.activated.emit(text)
        # print('got signal from', self.sender().text(), text)
        self._active_producers = []
        for a in self._actions:
            if a.isChecked():
                # print ('action', a.text(), 'is checked.')
                self._active_producers.append(a.text())
        self._owner.wireChoiceWorker(status=True, activeProducers=self._active_producers)
        return

    def name(self):
        return self._name




class recoBox(QtGui.QWidget):
    activated = QtCore.pyqtSignal(str)

    '''
    A widget class that contains the label and combo box.
    It also knows what to do when updating
    '''

    def __init__(self, owner, name, product, producers):
        super(recoBox, self).__init__()
        self._label = QtGui.QLabel()
        self._name = name
        self._label.setText(self._name.capitalize() + ": ")
        self._box = ComboBoxWithKeyConnect()
        self._box.setDuplicatesEnabled(False)
        self._box.activated[str].connect(self.emitSignal)
        self._product = product
        self._producers = producers
        self._current_producer = None
        self._stage = "all"
        if producers == None:
            self._box.addItem("--None--")
        else:
            self._box.addItem("--Select--")
            for producer in producers:
                self._box.addItem(producer.producer())

        self._box.connectOwnerKPE(owner.keyPressEvent)

        # This is the widget itself, so set it up
        self._layout = QtGui.QHBoxLayout()
        self._layout.addWidget(self._label)
        self._layout.addWidget(self._box)
        self.setLayout(self._layout)

    def selectStage(self, stage):

        # If no stage can draw this product, just return
        if self._producers is None:
            return
        else:
            self._box.clear()

            prod_list = []
            for prod in self._producers:
                if prod.stage() == stage or stage == 'all':
                    prod_list.append(prod.producer())

            if len(prod_list) > 0:
                self._box.addItem("--Select--")
                for _producer in prod_list:
                    self._box.addItem(_producer)
            else:
                self._box.addItem("--None--")

        self._box.setDuplicatesEnabled(False)


    def keyPressEvent(self, e):
        self._box.keyPressEvent(e)
        super(recoBox, self).keyPressEvent(e)

    def emitSignal(self, text):
        self.activated.emit(text)

    def productType(self):
        return self._product

    def productObj(self, producer, stage):
        if stage is None:
            stage = "all"
        for prod in self._producers:
            if prod.producer() == producer:
                if stage == "all" or prod.stage() == stage:
                    self._current_producer = producer
                    return prod
        self._current_producer = None
        return None

    def name(self):
        return self._name

    def currentProducer(self):
        return self._current_producer
