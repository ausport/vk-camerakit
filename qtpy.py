import sys
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QAction, QMainWindow, QHBoxLayout, QLabel, QGridLayout)
from PyQt5.QtGui import (QFont, QIcon)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon
from PIL.ImageQt import ImageQt

import VideoTools as tools


#http://zetcode.com/gui/pyqt5/firstprograms/
#http://zetcode.com/gui/pyqt5/eventssignals/
#https://pythonprogramminglanguage.com/pyqt5-video-widget/

class Annotator(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.mpv = tools.VideoObject("/Users/stuartmorgan/Desktop/Cam1a.mp4", target_resolution = (720, 1280))
        self.mpv.showDetails()
        self._position = 1
        im = self.mpv.getFrame(self._position)

        qim = ImageQt(im)
        pixmap = QPixmap.fromImage(qim)
        self._frames = self.mpv.frameCount

        self.videoWidget = QLabel()

        self.videoWidget.setMinimumHeight(720)
        self.videoWidget.setMinimumWidth(1280)
        self.videoWidget.setPixmap(pixmap)

        hbox = QHBoxLayout()
        hbox.addWidget(self.videoWidget)

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
        self.positionSlider.setRange(0, self._frames)
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
        layout.addLayout(hbox)
        layout.addLayout(controlLayout)

        self.setLayout(layout)


        self.setWindowTitle("QtPy Video Annotator")
        self.show()

        self.setFixedSize(self.size())



    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def step(self):
        if self._position < self._frames:
            self._position = self._position + 1
        self.setPosition(self._position)

    def setPosition(self, position):
        if position < self._frames:
            print("Set positions", position)
            # self.mediaPlayer.setPosition(position)
            im = self.mpv.getFrame(position)
            qim = ImageQt(im)
            pixmap = QPixmap.fromImage(qim)
            self.videoWidget.setPixmap(pixmap)
            self._position = position

    def mouseMoveEvent(self, e):

        x = e.x()
        y = e.y()

        text = "x: {0},  y: {1}".format(x, y)
        print(text)
        # self.label.setText(text)


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



if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Annotator()
    sys.exit(app.exec_())
