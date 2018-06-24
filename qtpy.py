import sys
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QAction, QMainWindow, QHBoxLayout, QLabel, QGridLayout)
from PyQt5.QtGui import (QFont, QIcon, QPainter, QPen)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon
from PIL.ImageQt import ImageQt

import VideoTools as tools
import random

#http://zetcode.com/gui/pyqt5/firstprograms/
#http://zetcode.com/gui/pyqt5/eventssignals/
#https://pythonprogramminglanguage.com/pyqt5-video-widget/

class VideoWindow(QWidget):

    def __init__(self):
        # super().__init__()
        QWidget.__init__(self)
        print("VideoWindow")
        print(self)
        self.initUI()



    def initUI(self):
        print("Creating Video Windows")

        self.mpv = tools.VideoObject("/Users/stuartmorgan/Desktop/Cam1a.mp4", target_resolution = (720, 1280))
        self.mpv.showDetails()
        self._position = 1
        self._frames = self.mpv.frameCount

        self.videoWidget = QLabel(self)
        self.videoWidget.mousePressEvent = self.video_mousePressEvent
        self.videoWidget.mouseMoveEvent = self.video_mouseMoveEvent
        self.videoWidget.mouseReleaseEvent = self.video_mouseReleaseEvent

        self.videoWidget.setMinimumHeight(720)
        self.videoWidget.setMinimumWidth(1280)


        im = self.mpv.getFrame(self._position)
        qim = ImageQt(im)
        pixmap = QPixmap.fromImage(qim)
        self.videoWidget.setPixmap(pixmap)



    def video_mousePressEvent(self, e):

        x = e.x()
        y = e.y()
        text = "x: {0},  y: {1}".format(x, y)
        print(text)
        self.drawPoints(self.qp)
        # self.label.setText(text)

    def video_mouseMoveEvent(self, e):

        x = e.x()
        y = e.y()
        text = "MOvingK x: {0},  y: {1}".format(x, y)
        print(text)
        # self.label.setText(text)
    def video_mouseReleaseEvent(self, e):

        x = e.x()
        y = e.y()
        text = "released at x: {0},  y: {1}".format(x, y)
        print(text)

    def paintEvent(self, e):
        print("doing this")
        im = self.mpv.getFrame(self._position)

        qim = ImageQt(im)
        pixmap = QPixmap.fromImage(qim)
        self.videoWidget.setPixmap(pixmap)

        painter = QPainter(self)
        pixmap = QPixmap.fromImage(qim)
        painter.drawPixmap(self.rect(), pixmap)

        pen = QPen(Qt.red, 3)
        painter.setPen(pen)
        painter.drawLine(10, 10, self.rect().width() -10 , 10)

    def closeEvent(self, event):
        print("Closing")
        self.mpv.close()

class Annotator(QWidget):

    def __init__(self):
        super().__init__()
        print("Annotator")
        print(self)

        self.initUI()


    def initUI(self):

        # self.mpv = tools.VideoObject("/Users/stuartmorgan/Desktop/Cam1a.mp4", target_resolution = (720, 1280))
        # self.mpv.showDetails()
        # self._position = 1
        # self._frames = self.mpv.frameCount
        #
        # self.videoWidget = QLabel(self)
        # self.videoWidget.mousePressEvent = self.video_mousePressEvent
        # self.videoWidget.mouseMoveEvent = self.video_mouseMoveEvent
        # self.videoWidget.mouseReleaseEvent = self.video_mouseReleaseEvent

        # self.videoWidget.setMinimumHeight(720)
        # self.videoWidget.setMinimumWidth(1280)
        # self.videoWidget.setMouseTracking(True)

        print("Will create VideoWindow")
        self.videoWidget = VideoWindow()
        print("Did create VideoWindow")

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.videoWidget)

        #Create buttons/sliders
        self.stepButton = QPushButton()
        self.stepButton.setEnabled(True)
        self.stepButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.stepButton.setToolTip('Step foward')
        self.stepButton.clicked.connect(self.step)

        # self.playButton.clicked.connect(self.play)
        self.nextButton = QPushButton()
        self.nextButton.setEnabled(True)
        self.nextButton.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.nextButton.setToolTip('Jump to next annotated frame')

        self.prevButton = QPushButton()
        self.prevButton.setEnabled(True)
        self.prevButton.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.prevButton.setToolTip('Jump to previous annotated frame')


        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, self.videoWidget._frames)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        #
        #
        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.stepButton)
        controlLayout.addWidget(self.prevButton)
        controlLayout.addWidget(self.nextButton)
        controlLayout.addWidget(self.positionSlider)
        #
        layout = QVBoxLayout(self)
        layout.addLayout(self.hbox)
        layout.addLayout(controlLayout)

        self.setLayout(layout)


        self.setWindowTitle("QtPy Video Annotator")
        self.show()

        print("here")
        # self.qp.begin(self.pixmap)

        self.setFixedSize(self.size())


    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def step(self):
        if self.videoWidget._position < self.videoWidget._frames:
            self.videoWidget._position = self.videoWidget._position + 1
        self.setPosition(self.videoWidget._position)

    def setPosition(self, position):
        if position < self.videoWidget._frames:
            print("Set positions", position)
            # # self.mediaPlayer.setPosition(position)
            # im = self.videoWidget.mpv.getFrame(position)
            # qim = ImageQt(im)
            # pixmap = QPixmap.fromImage(qim)
            # self.videoWidget.setPixmap(pixmap)
            # self.videoWidget._position = position

    # def video_mousePressEvent(self, e):
    #
    #     x = e.x()
    #     y = e.y()
    #     text = "x: {0},  y: {1}".format(x, y)
    #     print(text)
    #     self.drawPoints(self.qp)
    #     # self.label.setText(text)
    #
    # def video_mouseMoveEvent(self, e):
    #
    #     x = e.x()
    #     y = e.y()
    #     text = "MOvingK x: {0},  y: {1}".format(x, y)
    #     print(text)
    #     # self.label.setText(text)
    # def video_mouseReleaseEvent(self, e):
    #
    #     x = e.x()
    #     y = e.y()
    #     text = "released at x: {0},  y: {1}".format(x, y)
    #     print(text)


        # self.videoWidget.setPixmap(pixmap)
        # self.videoWidget.setPixmap(self.pixmap)

        # self.drawPoints(qp)
        # qp.end()

    def drawPoints(self, qp):
        print("Drawing")
        qp.setPen(Qt.red)
        size = self.size()

        for i in range(1000):
            x = random.randint(1, size.width()-1)
            y = random.randint(1, size.height()-1)
            qp.drawPoint(x, y)
        qp.drawRect(100, 100, 200, 200)
        # self.pixmap.drawRect(100, 100, 200, 200)

    # def closeEvent(self, event):
    #
    #     reply = QMessageBox.question(self, 'Message',
    #         "Are you sure to quit?", QMessageBox.Yes |
    #         QMessageBox.No, QMessageBox.No)
    #
    #     if reply == QMessageBox.Yes:
    #         event.accept()
    #     else:
    #         event.ignore()

    def __del__(self):
        print("Closing")


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Annotator()
    sys.exit(app.exec_())
