from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

class VerticalLabel(QtWidgets.QLabel):

    def __init__(self, *args):
        super().__init__(*args)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.translate(0, self.height())
        painter.rotate(-90)
        painter.drawText(0, self.width()/2, self.text())
        painter.end()


class ElidedLabel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumWidth(50)

    def paintEvent(self, event):
        # super().paintEvent(event);

        painter = QtGui.QPainter(self);
        fontMetrics = painter.fontMetrics();
        elidedLine = fontMetrics.elidedText(self.text(), QtCore.Qt.ElideMiddle, self.width());
        painter.drawText(QtCore.QPoint(0, fontMetrics.ascent()), elidedLine);
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


class MultiSelectionBox(QtWidgets.QToolButton):
    """ Allows users to select multiple checkable products from a drop-down button and displays the selected objects """
    activated = QtCore.pyqtSignal()
    _DEFAULT_STR = 'Select'

    def __init__(self, owner, name, products, default_products=None, mutually_exclusive=True):
        super().__init__()
        self._name = name
        self._owner = owner

        self.default_products = []
        if default_products is not None:
            self.default_products = default_products

        self._mutually_exclusive = mutually_exclusive

        self.setText(MultiSelectionBox._DEFAULT_STR)
        self._toolmenu = QtWidgets.QMenu(self)
        self._actions = []

        self.setMenu(self._toolmenu)
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)


        # set explicit minimum size to let the box shrink even if the label is
        # long. the overriden paint method adds ellipses to long strings
        # automatically
        self.setMinimumWidth(50)

        self.set_products(products)

    def set_products(self, products):
        self._toolmenu.clear()
        first = True
        if products is not None:
            group = QtWidgets.QActionGroup(self._toolmenu)
            for i, product in enumerate(products):
                action = self._toolmenu.addAction(product.full_name())
                action.setCheckable(True)
                if self._mutually_exclusive:
                    action.setActionGroup(group)
                if product in self.default_products and not self._mutually_exclusive:
                    action.setChecked(True)
                if first and self._mutually_exclusive:
                    action.setChecked(True)
                    first = False
                action.triggered.connect(self.on_action_triggered)
                self._actions.append(action)
        else:
            action = self._toolmenu.addAction(f"No {self._name} available to draw.")
            action.setCheckable(False)
        self.setText(self.display_name())

    def on_action_triggered(self):
        self.setText(self.display_name())
        self.activated.emit()

    def display_name(self):
        selected = [a for a in self._toolmenu.actions() if a.isChecked()]
        n = len(selected)
        if n == 0:
            return MultiSelectionBox._DEFAULT_STR
        if n == 1:
            return selected[0].text()
        return f"{n} selected"
    
    @property
    def name(self):
        return self._name

    def selected_products(self):
        return [a.text() for a in self._toolmenu.actions() if a.isChecked()]

    def select(self, product):
        avail_products = [a.text() for a in self._toolmenu.actions()]
        if product in avail_products:
            self._toolmenu.actions()[avail_products.index(product)].setChecked(True)

    def paintEvent(self, event):
        '''
        Override to elide the text.
        See https://doc.qt.io/qtforpython-6.2/overviews/qtwidgets-widgets-elidedlabel-example.html
        https://stackoverflow.com/questions/41360618/qcombobox-elided-text-on-selected-item
        The -10 corrects for the pixels associated with the menu button icon
        '''

        opt = QtWidgets.QStyleOptionToolButton()
        self.initStyleOption(opt);
        opt.text = ""
        painter = QtWidgets.QStylePainter(self)
        painter.drawComplexControl(QtWidgets.QStyle.CC_ToolButton, opt);
        font_metrics = painter.fontMetrics()
        text_rect = self.style().subControlRect(QtWidgets.QStyle.CC_ToolButton, opt, QtWidgets.QStyle.SC_ToolButton, self)
        opt.text = font_metrics.elidedText(self.text(), QtCore.Qt.ElideRight, text_rect.width() - 15)
        painter.drawControl(QtWidgets.QStyle.CE_ToolButtonLabel, opt)


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
        self.set_product_and_producers(self._product, self._producers)

        self.connectOwnerKPE(owner.keyPressEvent)

    def set_product_and_producers(self, product, producers):
        self.clear()
        self._product = product
        self._producers = producers
        
        if producers == None:
            self.addItem("--None--")
        else:
            self.addItem("--Select--")
            for producer in producers:
                self.addItem(producer.producer())

    def selectStage(self, stage):
        # If no stage can draw this product, just return
        if self._producers is None:
            return
        
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


class MovablePixmapItem(QtWidgets.QGraphicsPixmapItem):
    """ GraphicsPixmapItem which can be clicked & dragged """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        self._border_pen = QtGui.QPen(
            QtCore.Qt.yellow, 10,
            QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin
        )

        self._border = QtWidgets.QGraphicsRectItem(self.boundingRect())
        self._border.setPen(self._border_pen)

    def hoverEnterEvent(self, ev):
        super().hoverEnterEvent(ev)
        self.scene().addItem(self._border)
        self._border.setPos(self.pos())

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        self._border.setPos(self.pos())

    def hoverLeaveEvent(self, ev):
        super().hoverLeaveEvent(ev)
        self.scene().removeItem(self._border)

    def setScale(self, scale):
        super().setScale(scale)
        self._border.setScale(scale)


class MovableScaleBar(pg.ScaleBar):
    '''
    scale bar which can be moved around. The inner bar highlights when the
    mouse hovers over it
    '''
    class HighlightBar(QtWidgets.QGraphicsRectItem):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setAcceptHoverEvents(True)
            self._highlight_brush = QtGui.QBrush(QtCore.Qt.yellow, 1)
            self._brush = None

        def setDefaultBrush(self, brush):
            super().setBrush(brush)
            self._brush = brush

        def hoverEnterEvent(self, ev):
            super().hoverEnterEvent(ev)
            self.setBrush(self._highlight_brush)

        def hoverLeaveEvent(self, ev):
            super().hoverLeaveEvent(ev)
            self.setBrush(self._brush)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        
        self.bar = MovableScaleBar.HighlightBar()
        self.bar.setPen(self.pen)
        self.bar.setDefaultBrush(self.brush)
        self.bar.setParentItem(self)

        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)

        self._last_pos = None

    def anchor(self, itemPos, parentPos, offset=(0,0)):
        super().anchor(itemPos, parentPos, offset)
        self.offset = offset

    def boundingRect(self):
        return self.bar.rect()

    def setUnits(self, scale=1, suffix=''):
        self.text.setText(pg.functions.siFormat(self.size * scale, suffix=suffix))

    def setFont(self, font):
        self.text.setFont(font)


class MovableLabel(pg.TextItem):
    """ GraphicsPixmapItem which can be clicked & dragged """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        self._border_pen = QtGui.QPen(
            QtCore.Qt.yellow, 10,
            QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin
        )

        # self._border = QtWidgets.QGraphicsRectItem(self.boundingRect())
        # self._border.setPen(self._border_pen)

    # def hoverEnterEvent(self, ev):
    #     super().hoverEnterEvent(ev)
    #     self.scene().addItem(self._border)
    #     self._border.setPos(self.pos())

    # def mouseMoveEvent(self, ev):
    #     super().mouseMoveEvent(ev)
    #     self._border.setPos(self.pos())

    # def hoverLeaveEvent(self, ev):
    #     super().hoverLeaveEvent(ev)
    #     self.scene().removeItem(self._border)

    # def setScale(self, scale):
    #     super().setScale(scale)
    #     self._border.setScale(scale)


