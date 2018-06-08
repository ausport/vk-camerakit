import sys, math
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
#https://www.learnopencv.com/homography-examples-using-opencv-python-c/

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
        super(ImageViewer, self).mousePressEvent(event)

    def SceneClicked(self, pos):
        # Pass local (scene) coordinates to ImageClicked()
        if self._Image.isUnderMouse():
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
        self.btnAddCorrespondances = QToolButton(self)
        self.btnAddCorrespondances.setText('Add Correspondance')
        self.btnAddCorrespondances.clicked.connect(self.pixInfo)
        self.btnComputeHomograhy = QToolButton(self)
        self.btnComputeHomograhy.setText('Compute Homograhy')
        self.btnComputeHomograhy.clicked.connect(self.computeHomograhy)
        self.editImageCoordsInfo = QLineEdit(self)
        self.editImageCoordsInfo.setReadOnly(True)
        self.editModelCoords = QLineEdit(self)
        self.editModelCoords.setReadOnly(False)
        self.editModelCoords.returnPressed.connect(self.addCorrespondances)
        self.listCorrespondances = QListWidget()
        self.viewer.ImageClicked.connect(self.ImageClicked)
        self._mylastImagePairs = {0,0}
        # Arrange layout
        VBlayout = QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        VBlayout.addWidget(self.listCorrespondances)

        HBlayout = QHBoxLayout()
        HBlayout.setAlignment(Qt.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        HBlayout.addWidget(self.btnAddCorrespondances)
        HBlayout.addWidget(self.editImageCoordsInfo)
        HBlayout.addWidget(self.editModelCoords)
        HBlayout.addWidget(self.btnComputeHomograhy)
        VBlayout.addLayout(HBlayout)

        #Initial data:
        #x-image, y-image, x-model, y-model
        self._my_correspondances = []
        self._my_correspondances.append({'cx':460, 'cy':223, 'rx': 0, 'ry': 0})
        self._my_correspondances.append({'cx':1245, 'cy':454, 'rx': 25, 'ry': 25})
        self._my_correspondances.append({'cx':1152, 'cy':125, 'rx': 25, 'ry': -25})
        self._my_correspondances.append({'cx':541, 'cy':101, 'rx': 0, 'ry': -25})
        print(self._my_correspondances)

        for c in self._my_correspondances:
            self.listCorrespondances.addItem("{0}, {1} ,{2}, {3}".format(c['cx'], c['cy'], c['rx'], c['ry']))

    def loadImage(self):
        self.viewer.setImage(QPixmap("./shot0001.png"))
        #Draw point
        pen = QPen(Qt.red)
        brush = QBrush(Qt.yellow)
        for c in self._my_correspondances:
            self.viewer._scene.addEllipse(c['cx']-3, c['cy']-3, 6, 6, pen, brush)

    def pixInfo(self):
        self.viewer.toggleDragMode()

    def ImageClicked(self, pos):
        if self.viewer.dragMode()  == QGraphicsView.NoDrag:
            self.editImageCoordsInfo.setText('%d, %d' % (pos.x(), pos.y()))

            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.viewer._scene.addEllipse(pos.x()-3, pos.y()-3, 6, 6, pen, brush)
            self.viewer.toggleDragMode()
            self._mylastImagePairs = {pos.x(), pos.y()}
            self.editModelCoords.setStyleSheet("background-color: rgb(0, 255, 0);")

    def addCorrespondances(self):
        #Verify correct entry (x, y)
        inputNumber = self.editModelCoords.text()
        try:
            (xval,yval) = [int(s) for s in inputNumber.split(',')]
            print(xval,yval)
            self.editModelCoords.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.listCorrespondances.addItem("{0}, {1} ,{2}, {3}".format(list(self._mylastImagePairs)[0], list(self._mylastImagePairs)[1], xval,yval))
            # item = QListWidgetItem("Item %i" % i)

        except Exception as e:
            print(e)
            print("Please select a number, `{0}` isn't valid!".format(inputNumber))
            return

    def computeHomograhy(self):
        '''
        pts_src and pts_dst are numpy arrays of points
        in source and destination images. We need at least
        4 corresponding points.
        '''
        pts_src = np.array([[460, 223], [1245, 454], [1152, 125],[541, 101]])
        pts_dst = np.array([[250, 250], [500, 500], [500, 0],[250, 0]])
        h, status = cv2.findHomography(pts_src, pts_dst)

        print(h)
        '''
        The calculated homography can be used to warp
        the source image to destination. Size is the
        size (width,height) of im_dst
        '''
        im_src = cv2.imread("./shot0001.png")
        # Warp source image to destination based on homography
        # im_out = cv2.warpPerspective(im_src, h, (im_src.shape[1],im_src.shape[0]))
        im_out = cv2.warpPerspective(im_src, h, (1920,1080))

        # Display images
        # cv2.imshow("Source Image", im_src)
        # cv2.imshow("Destination Image", im_dst)
        # cv2.imshow("Warped Source Image", im_out)
        height, width, channel = im_out.shape
        bytesPerLine = 3 * width
        qImg = QImage(im_out.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.viewer.setImage(QPixmap(qImg))

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    sys.exit(app.exec_())
