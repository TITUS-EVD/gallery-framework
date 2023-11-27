from PyQt5 import QtGui, QtCore, QtWidgets

class VerticalLabel(QtWidgets.QLabel):

    def __init__(self, *args):
        super().__init__(*args)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.translate(0, self.height())
        painter.rotate(-90)
        painter.drawText(0, self.width()/2, self.text())
        painter.end()

class ComboBoxWithKeyConnect(QtWidgets.QComboBox):

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

class waveformBox(QtWidgets.QToolButton):
    activated = QtCore.pyqtSignal(str)

    def __init__(self, owner, name, products, default_products=[]):
        super(waveformBox, self).__init__()
        self._name = name
        self._owner = owner

        self.setText('Select')
        self._toolmenu = QtWidgets.QMenu(self)
        self._actions = []
        if products is not None:
            for i, product in enumerate(products):
                action = self._toolmenu.addAction(product.full_name())
                action.setCheckable(True)
                if product in default_products:
                    action.setChecked(True)
                action.triggered.connect(self.emitSignal)
                self._actions.append(action)
        else:
            action = self._toolmenu.addAction("No " + name + " available to draw.")
            action.setCheckable(False)

        self.setMenu(self._toolmenu)
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)


    def emitSignal(self, text):
        # self.activated.emit(text)
        # print('got signal from', self.sender().text(), text)
        self._active_producers = []
        for a in self._actions:
            if a.isChecked():
                # print ('action', a.text(), 'is checked.')
                self._active_producers.append(a.text())
        self._owner.change_wire_choice(status=True, activeProducers=self._active_producers)
        return

    def name(self):
        return self._name




class recoBox(ComboBoxWithKeyConnect):
    '''
    A widget class that contains the label and combo box.
    It also knows what to do when updating
    '''

    def __init__(self, owner, name, product, producers):
        super(recoBox, self).__init__()
        self._name = name
        self.setDuplicatesEnabled(False)
        self._product = product
        self._producers = producers
        self._current_producer = None
        self._stage = "all"
        if producers == None:
            self.addItem("--None--")
        else:
            self.addItem("--Select--")
            for producer in producers:
                self.addItem(producer.producer())

        self.connectOwnerKPE(owner.keyPressEvent)

    def selectStage(self, stage):
        # If no stage can draw this product, just return
        if self._producers is None:
            return
        else:
            self.clear()

            prod_list = []
            for prod in self._producers:
                if prod.stage() == stage or stage == 'all':
                    prod_list.append(prod.producer())

            if len(prod_list) > 0:
                self.addItem("--Select--")
                for _producer in prod_list:
                    self.addItem(_producer)
            else:
                self.addItem("--None--")

        self.setDuplicatesEnabled(False)

    def productType(self):
        return self._product

    def productObj(self, producer, stage):
        if self._producers is None:
            return None

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
