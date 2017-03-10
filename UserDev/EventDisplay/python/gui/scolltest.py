from PyQt4 import QtCore, QtGui

app = QtGui.QApplication([])

win = QtGui.QDialog()
win.resize(300,300)
layout = QtGui.QVBoxLayout(win)
scroll = QtGui.QScrollArea()
scroll.setWidgetResizable(True)
layout.addWidget(scroll)

scrollContents = QtGui.QWidget()
layout = QtGui.QVBoxLayout(scrollContents)
scroll.setWidget(scrollContents)

pix = QtGui.QPixmap("image.png")

def createImage():
    label = QtGui.QLabel()
    label.setPixmap(pix)
    layout.addWidget(label)

t = QtCore.QTimer(win)
t.timeout.connect(createImage)
t.start(500)

win.show()
win.raise_()
app.exec_()