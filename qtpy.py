import sys
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QAction, QMainWindow, QHBoxLayout, QLabel)
from PyQt5.QtGui import (QFont, QIcon)
from PyQt5.QtGui import QPixmap

from PIL.ImageQt import ImageQt

import VideoTools as tools


#http://zetcode.com/gui/pyqt5/firstprograms/
#http://zetcode.com/gui/pyqt5/eventssignals/

class Annotator(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        QToolTip.setFont(QFont('SansSerif', 10))

        self.setToolTip('This is a <b>QWidget</b> widget')

        btn = QPushButton('Button', self)
        btn.setToolTip('This is a <b>QPushButton</b> widget')
        btn.resize(btn.sizeHint())
        btn.move(50, 50)

        btn2 = QPushButton('Save and Quit', self)
        btn2.setToolTip('This is a <b>QPushButton</b> widget')
        btn2.resize(btn2.sizeHint())
        btn2.move(150, 50)


        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(app.quit)

        hbox = QHBoxLayout(self)

        mpv = tools.VideoObject("/Users/stuartmorgan/Desktop/Cam1a.mp4", target_resolution = (720, 1280))
        mpv.showDetails()
        im = mpv.getFrame(1)
        mpv.close()

        qim = ImageQt(im)
        pixmap = QPixmap.fromImage(qim)

        lbl = QLabel(self)

        lbl.setMinimumHeight(720)
        lbl.setMinimumWidth(1280)
        lbl.setPixmap(pixmap)

        hbox.addWidget(lbl)
        self.setLayout(hbox)

        self.setGeometry(300, 300, 500, 600)
        self.setWindowTitle('Tooltips')
        self.show()

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes |
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()



if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Annotator()
    sys.exit(app.exec_())
