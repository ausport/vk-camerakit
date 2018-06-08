import sys, math
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PIL import Image, ImageDraw, ImageFont

class GraphicsScene(QGraphicsScene):
    #Create signal exporting QPointF position.
    SceneClicked = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

        self.setSceneRect(-100, -100, 200, 200)
        self.opt = ""

    def setOption(self, opt):
        self.opt = opt

    def mousePressEvent(self, event):
        pen = QPen(Qt.red)
        brush = QBrush(Qt.yellow)
        x = event.scenePos().x()
        y = event.scenePos().y()
        if True:
            self.addEllipse(x-3, y-3, 6, 6, pen, brush)
        elif self.opt == "Select":
            pass
            # print(x, y)

        # #Emit the signal
        self.SceneClicked.emit(QPointF(event.scenePos()))

class ImageViewer(QGraphicsView):
    ImageClicked = pyqtSignal(QPoint)

    def __init__(self, parent):
        super(ImageViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = GraphicsScene()
        self._Image = QGraphicsPixmapItem()
        self._scene.addItem(self._Image)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)

        #Connect the signal emitted by the GraphicsScene mousePressEvent to relay event
        self._scene.SceneClicked.connect(self.SceneClicked)


    def hasImage(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QRectF(self._Image.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasImage():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setImage(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._Image.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._Image.setPixmap(QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasImage():
            if event.angleDelta().y() > 0:
                factor = 1.1
                self._zoom += 1
            else:
                factor = 0.9
                self._zoom -= 1

            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
        elif not self._Image.pixmap().isNull():
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._Image.isUnderMouse():
            self.ImageClicked.emit(QPoint(event.pos()))

        super(ImageViewer, self).mousePressEvent(event)

    def SceneClicked(self, pos):
        # Pass local (scene) coordinates to ImageClicked()
        self.ImageClicked.emit(pos.toPoint())


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.viewer = ImageViewer(self)
        # 'Load image' button
        self.btnLoad = QToolButton(self)
        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.loadImage)
        # Button to change from drag/pan to getting pixel info
        self.btnPixInfo = QToolButton(self)
        self.btnPixInfo.setText('Correspondance Mode is OFF')
        self.btnPixInfo.clicked.connect(self.pixInfo)
        self.editPixInfo = QLineEdit(self)
        self.editPixInfo.setReadOnly(True)
        self.viewer.ImageClicked.connect(self.ImageClicked)

        # Arrange layout
        VBlayout = QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        HBlayout = QHBoxLayout()
        HBlayout.setAlignment(Qt.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        HBlayout.addWidget(self.btnPixInfo)
        HBlayout.addWidget(self.editPixInfo)
        VBlayout.addLayout(HBlayout)


    def loadImage(self):
        self.viewer.setImage(QPixmap("./shot0001.png"))

    def pixInfo(self):
        self.viewer.toggleDragMode()
        if self.viewer.dragMode()  == QGraphicsView.NoDrag:
            self.btnPixInfo.setText('Correspondance Mode is ON')
        else:
            self.btnPixInfo.setText('Correspondance Mode is OFF')

    def ImageClicked(self, pos):
        if self.viewer.dragMode()  == QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

        # if self.viewer.dragMode()  == QGraphicsView.NoDrag:
        #     self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    sys.exit(app.exec_())
