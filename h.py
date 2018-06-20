import sys, math
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
#https://www.learnopencv.com/homography-examples-using-opencv-python-c/


# TODO: Add distortion slider.
# TODO: Fix corresponances and include auto entry.


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
            self.setDragMode(QGraphicsView.NoDrag)
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
        else:
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
        self.surface = ImageViewer(self)
        # 'Load image' button
        self.btnLoad = QToolButton(self)
        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.loadImage)

        self.cboSurfaces = QComboBox()
        for s in ("issia", "ncaacourt", "ncaafield", "netball", "hockey", "rugby", "tennis", "pool"):
            self.cboSurfaces.addItem(s)
        self.cboSurfaces.setCurrentText("tennis")
        self.cboSurfaces.currentIndexChanged.connect(self.updateImageSet)

        # Button to change from drag/pan to getting pixel info
        self.btnAddCorrespondances = QToolButton(self)
        self.btnAddCorrespondances.setText('Add Correspondance')
        self.btnAddCorrespondances.clicked.connect(self.addCorrespondances)
        self.btnComputeHomograhy = QToolButton(self)
        self.btnComputeHomograhy.setText('Compute Homograhy')
        self.btnComputeHomograhy.clicked.connect(self.computeHomograhy)
        self.editImageCoordsInfo = QLineEdit(self)
        self.editImageCoordsInfo.setReadOnly(True)
        self.sliderFocalLength = QSlider(Qt.Horizontal)
        self.sliderFocalLength.setMinimum(0)
        self.sliderFocalLength.setMaximum(50)
        self.sliderFocalLength.setValue(10)
        self.sliderFocalLength.setTickPosition(QSlider.TicksBelow)
        self.sliderFocalLength.setTickInterval(1)
        self.sliderFocalLength.valueChanged.connect(self.updateFocalLength)

        self.listCorrespondances = QListWidget()
        self.viewer.ImageClicked.connect(self.ImageClicked)
        self.surface.ImageClicked.connect(self.SurfaceClicked)
        self._mylastImagePairs = {0,0}
        self.addingCorrespondancesEnabled = False
        self.surfaceDimensions = None

        #Swimming defaults
        self.model_width = 50
        self.model_height = 25
        self.model_offset_x = 1
        self.model_offset_y = 1
        #Scaling factor required to convert from real world in meters to surface pixels.
        self.model_scale = 10


        # Camera properties
        self._myHomography = None
        self._focal_length = 10.8
        self._myCameraMatrix = None
        self._myDistortionMatrix = None
        self._myRotationVector = None
        self._myTranslationVector = None

        # Arrange layout
        VBlayout = QVBoxLayout(self)
        HB_images_layout = QHBoxLayout()

        HB_images_layout.addWidget(self.viewer)
        HB_images_layout.addWidget(self.surface)
        VBlayout.addLayout(HB_images_layout)

        HBlayout = QHBoxLayout()
        HBlayout.setAlignment(Qt.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        HBlayout.addWidget(self.btnAddCorrespondances)
        HBlayout.addWidget(self.editImageCoordsInfo)
        # HBlayout.addWidget(self.editModelCoords)
        HBlayout.addWidget(self.cboSurfaces)
        HBlayout.addWidget(self.sliderFocalLength)
        HBlayout.addWidget(self.btnComputeHomograhy)
        VBlayout.addLayout(HBlayout)


    def resetControls(self):

        #Abort corresponances
        self._mylastImagePairs = {0,0}
        self._mylastSurfacePairs = {0,0}
        self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.btnAddCorrespondances.setStyleSheet("background-color: None")
        self.addingCorrespondancesEnabled = False
        self.viewer.setDragMode(QGraphicsView.NoDrag)
        self.surface.setDragMode(QGraphicsView.NoDrag)


    def keyPressEvent(self, event):

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_Escape:
                #Abort corresponances
                self.resetControls()
                return

            if self.viewer._empty or self.surface._empty:
                return

            if event.key() == Qt.Key_Space:
                self.viewer.toggleDragMode()
                self.surface.toggleDragMode()

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == Qt.Key_Space:
                self.viewer.toggleDragMode()
                self.surface.toggleDragMode()

    def loadSurface(self):

        px = QPixmap("./Surfaces/{:s}.png".format(self.cboSurfaces.currentText()))
        self.surface.setImage(px)
        self.surfaceDimensions = px.size()
        print("Loading surface:", self.cboSurfaces.currentText(), self.surfaceDimensions)

    def loadImage(self):
        self.viewer.setImage(QPixmap("./Images/{:s}.png".format(self.cboSurfaces.currentText())))

    def updateImageSet(self):
        self.loadImage()
        self.loadSurface()

    def pixInfo(self):
        # self.viewer.toggleDragMode()
        if self.addingCorrespondancesEnabled:
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 100, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

    def ImageClicked(self, pos):
        print("Image")
        if self.viewer.dragMode()  == QGraphicsView.NoDrag and self.addingCorrespondancesEnabled == True:
            # self.editImageCoordsInfo.setText('%d, %d' % (pos.x(), pos.y()))
            print(pos.x(), pos.y())
            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.viewer._scene.addEllipse(pos.x()-3, pos.y()-3, 6, 6, pen, brush)
            self.viewer.toggleDragMode()
            self._mylastImagePairs = {pos.x(), pos.y()}
            # self.editModelCoords.setStyleSheet("background-color: rgb(0, 255, 0);")
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 100, 30)))
            print("_mylastImagePairs:", self._mylastImagePairs)


    def _drawImageSpaceDetection(self, pos):

# TODO: store homography gloablly for reference here.
        return
        # Render reference point annotation.
        r = 5
        yellow = Qt.yellow
        pen = QPen(Qt.red)
        brush = QBrush(QColor(255, 255, 0, 100))

        poly = QPolygonF()
        x, y = pos.x(), pos.y()
        poly_points = np.array([])
        #
        # # Compute inverse of 2D homography
        # print("**", _myHomography)
        #
        # val, H = cv2.invert(_myHomography)
        #
        for i in range(1, 33):
            #These points are in world coordinates.
            _x = x + (r * math.cos(2 * math.pi * i / 32))
            _y = y + (r * math.sin(2 * math.pi * i / 32))

                # ground_point = np.array([[[world_point[0], world_point[1]]]], dtype='float32')
                # ground_point = cv2.perspectiveTransform(ground_point, H)
                # ref_point = np.array([[[world_point[0], world_point[1], -10]]], dtype='float32')
                # (ref_point, jacobian) = cv2.projectPoints(ref_point, rotation_vector, translation_vector, camera_matrix, distortion_matrix)
                # # Render vertical
                # im_src = cv2.line(im_src, tuple(ground_point.ravel()), tuple(ref_point.ravel()), (0,255,255), 2)


            #Convert to image coordinates.
            axis = np.float32([[_x, _y]]).reshape(-1,2)
            imgpts = cv2.perspectiveTransform(axis, H)

            #Draw the points in a circle in perspective.
            (xx, yy) = tuple(imgpts[0].ravel())

            poly_points = np.append(poly_points, [xx, yy])

            _p = QPointF(xx,yy)
            poly.append(QPointF(xx,yy))

        self.viewer._scene.addPolygon(poly, pen, brush)

        #Render image-space point
        # axis = np.float32([[pos.x(),pos.y()]]).reshape(-1,3)
        axis = np.float32([[pos.x(),pos.y(),0], [pos.x(),pos.y(),-20]]).reshape(-1,3)
        (imgpts, jacobian) = cv2.projectPoints(axis, self._myRotationVector, self._myTranslationVector, self._myCameraMatrix, self._myDistortionMatrix)
        (x, y) = tuple(imgpts[0].ravel())
        (xx, yy) = tuple(imgpts[1].ravel())
        self.viewer._scene.addLine(xx, yy, x, y, pen)


    def SurfaceClicked(self, pos):
        print(self._myHomography.__class__.__name__)

        if not self._myHomography.__class__.__name__ == "NoneType":
            print("ok")
            print(pos.x(), pos.y())
            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.surface._scene.addEllipse(pos.x()-3, pos.y()-3, 6, 6, pen, brush)
            self.surface.toggleDragMode()
            self._mylastSurfacePairs = {pos.x(), pos.y()}

            self._drawImageSpaceDetection(pos)


        if self.surface.dragMode()  == QGraphicsView.NoDrag and self.addingCorrespondancesEnabled == True:
            # self.editImageCoordsInfo.setText('%d, %d' % (pos.x(), pos.y()))
            print(pos.x(), pos.y())
            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.surface._scene.addEllipse(pos.x()-3, pos.y()-3, 6, 6, pen, brush)
            self.surface.toggleDragMode()
            self._mylastSurfacePairs = {pos.x(), pos.y()}
            # self.editModelCoords.setStyleSheet("background-color: rgb(0, 255, 0);")
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            print("_mylastSurfacePairs:", self._mylastSurfacePairs)

            self.editImageCoordsInfo.setText("Image x:{0}, y:{1} : Surface x:{2}, y:{3}".format(list(self._mylastImagePairs)[0], list(self._mylastImagePairs)[1], list(self._mylastSurfacePairs)[0], list(self._mylastSurfacePairs)[1]))

            #Save corresponances
            self.resetControls()

    def addCorrespondances(self):
        #1. Highlight image space.
        if not self.addingCorrespondancesEnabled:
            self.addingCorrespondancesEnabled = True
            self.btnAddCorrespondances.setStyleSheet("background-color: green")

    def _draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img


    def updateFocalLength(self):
        self._focal_length = self.sliderFocalLength.value()
        self.computeHomograhy()

    def computeHomograhy(self):
        '''
        image_points and surface_points are numpy arrays of points
        in source and destination images. We need at least
        4 corresponding points.
        '''

        if self.cboSurfaces.currentText() == "pool":
            #Pool
            image_points = np.array([(832, 889), (155, 1394), (3046, 887),(3695, 1412)], dtype='float32')
            surface_points = np.array([ (10, 10, 0), (10, 260, 0), (510, 10, 0), (510, 260, 0) ], dtype='float32')
            self.model_width = 50
            self.model_height = 25
            self.model_offset_x = 1
            self.model_offset_y = 1
            #Scaling factor required to convert from real world in meters to surface pixels.
            self.model_scale = 10


        elif self.cboSurfaces.currentText() == "tennis":
            #Tennis
            image_points = np.array([(67, 293), (484, 288), (353, 157),(230, 158)], dtype='float32')
            surface_points = np.array([ (157, 102, 0), (157, 580, 0), (1343, 580, 0), (1343, 102, 0) ], dtype='float32')
            self.model_width = 30
            self.model_height = 15
            self.model_offset_x = 1
            self.model_offset_y = 1
            #Scaling factor required to convert from real world in meters to surface pixels.
            self.model_scale = 50


        # Compute 2D homography
        h, status = cv2.findHomography(image_points, surface_points)
        self._myHomography = h

        '''
        The calculated homography can be used to warp
        the source image to destination. Size is the
        size (width,height) of im_dst
        '''
        im_src = cv2.imread("./Images/{:s}.png".format(self.cboSurfaces.currentText()))

        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(im_src, h, (self.surfaceDimensions.width(),self.surfaceDimensions.height()))

        # Display images in QT
        height, width, channel = im_out.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB, im_out)
        qImg = QImage(im_out.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.surface.setImage(QPixmap(qImg))

        self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

        # Render pool boundaries.
        cv2.line(im_src, ( int(image_points[0][0]), int(image_points[0][1])), ( int(image_points[1][0]), int(image_points[1][1])), (255,255,0), 1)
        cv2.line(im_src, ( int(image_points[2][0]), int(image_points[2][1])), ( int(image_points[1][0]), int(image_points[1][1])), (255,0,255), 1)
        cv2.line(im_src, ( int(image_points[2][0]), int(image_points[2][1])), ( int(image_points[3][0]), int(image_points[3][1])), (0,255,0), 1)
        cv2.line(im_src, ( int(image_points[0][0]), int(image_points[0][1])), ( int(image_points[3][0]), int(image_points[3][1])), (0,255,255), 1)


        # Compute inverse of 2D homography
        val, H = cv2.invert(h)

        # Estimate naive camera intrinsics (camera matrix)
        print("Image dimensions :\n {0}".format(im_src.shape))
        h, w = im_src.shape[:2]
        fx = 0.5 + self._focal_length / 50.0
        camera_matrix = np.float64([[fx*w, 0, 0.5*(w-1)],
                        [0, fx*w, 0.5*(h-1)],
                        [0.0,0.0,      1.0]])

        print ("Camera Matrix {0}:\n {1}".format(self._focal_length, camera_matrix))

        # ...assuming no lens distortion
        distortion_matrix = np.zeros((4,1))
        print ("Distortion Matrix :\n {0}".format(distortion_matrix))

        #NB Generate square calibration corresponances using existing homography.
        # The problem is how to learn the camera pose, which we need to estimate a 3D camera coordinate system.
        # We need a) the camera instrinics, such as focal length and distortion, and b) camera extrinsics, rotation and translation.
        # Normally we would like to use a checkerboard for calibrating the camera, and deriving the camera instrinics.
        # OpenCV has lots of established medthods for doing this - such as cv2.CameraCalibration().
        # That's not very convenient for ad-hoc camera calibration, but there are some plausible workarounds.
        # Since we can easily estimate a 2D homography using points corresponding between the image and world coordinate systems,
        # that should give us a relible planar scene context to solve the camera extrinsics.
        # First we estimate naively the focal point (see camera_matrix above), and we assume no distortion(!) *more on that.
        # So, here we use a grid of equi-spaced points in world coordinates, and use the inverse homography to reliably retrieve the
        # (x,y) loation of the corresponding points in camera coordinates.  The perspective embedded in the grid-based camera coordinates should
        # approximate the same points learned from a checkerboard, and we use the cv2.solvePnP() function the give us the extrinsics: rotation and translation.
        # At this point, if we create a grid across the entire image space, the sover does it's best least-suqares approximation of the
        # extrinsics, bu we find when we project points back to the image space that the perspective model fails at a rate proportional to the
        # distance from the center of the image.  I suspect this is because of our previous assumption of zero camera distortion.
        # A hacky workaround is to use a local set of ground truth coordinates, and use the cv2.solvePnP() function
        # to derive compute quasi-extrinsic parameters.  In other words, let w(x,y,z) be the 3D coordinates in world space that we
        # want to project into 2D camera space c(x,y).  We take w(x,y,0), and build a 1m x 1m grid surrounding that point (ensuring we don't breach the real camera bounds).
        # For each point in the 1x1m grid we use the inverse homography transform and directly compute the 2D image coordinates, from which we can compute local
        # camera extrinsics.  Obviously this is a hack, and it would be nice to have more accurate global extrinsics parameters, but we would otherwise
        # need some way of computing the camera distortion accurately.

        _global_calibration = True

        if _global_calibration:
            world_points = np.zeros((self.model_width*self.model_height,3), np.float32)
            world_points[:,:2] = np.mgrid[self.model_offset_x:self.model_width+self.model_offset_x,self.model_offset_y:self.model_height+self.model_offset_y].T.reshape(-1,2)*self.model_scale #convert to meters (scale is 1:10)
            camera_points = np.zeros((self.model_width*self.model_height,2), np.float32)
        else:
            # Manually set world point calibration for local extrinsics.

            # Top left of pool
            world_points = np.float32([[10,10,0], [60,10,0], [10,60,0],  [60,60,0]])
            camera_points = np.zeros((2*2,2), np.float32)

            # Bottom right of pool space
            # world_points = np.float32([[510,250,0], [500,250,0], [510,240,0],  [500,240,0]])
            # camera_points = np.zeros((2*2,2), np.float32)


        # Estimate image point coordinates on the ground plane for world point calibration markers.
        i = 0
        for world_point in world_points:
            #Remove z-axis (assumed to be zero)
            point = np.array([[[world_point[0], world_point[1]]]], dtype='float32')
            #Get image coordinates for world coordinate point (x,y).
            camera_points[i] = cv2.perspectiveTransform(point, H)
            i = i + 1


        # Solve rotation and translation matrices
        (_, rotation_vector, translation_vector) = cv2.solvePnP(world_points, camera_points, camera_matrix, distortion_matrix)
        print ("Rotation Matrix :\n {0}".format(rotation_vector))
        print ("Translation Matrix :\n {0}".format(translation_vector))

        # Retain matrices
        self._myHomography = h
        self._myCameraMatrix = camera_matrix
        self._myDistortionMatrix = distortion_matrix
        self._myRotationVector = rotation_vector
        self._myTranslationVector = translation_vector

        # Top left of pool space
        axis = np.float32([[60,10,0], [10,60,0], [10,10,-50]]).reshape(-1,3)
        # Bottom right of pool space
        # axis = np.float32([[500,250,0], [510,240,0], [510,250,-50]]).reshape(-1,3)

        # Render reference point annotation.
        (imgpts, jacobian) = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, distortion_matrix)
        im_src = self._draw(im_src,camera_points,imgpts)


        if True:
            # This code demonstrates the problem with ignoring instrinsic camera distortion.
            # It should take the inverse homography image points (reliable), and project in the z-plane
            # using the camera extrinsics (rotation/translation) solved above using cv2.solvePnP().
            # If the 3D image coordinate system is accurate, the vertical yellow lines should be
            # generally consistent with normal perspective.
            # By switching between different calibration routines (above), we can see the effects of
            # poor or non-local calibration.
            # For instance, if we use a local calibration grid above, ~10m around the origin (10,10), then the
            # yellow verticals around that region are good, but the non-local distortion corrupts the 3D perspective.
            # Using a global calibration grid (i.e. a grid over the entire model space), then only the points near the
            # center of the image are in proper perspective.

            world_points = np.zeros((self.model_width*self.model_height,3), np.float32)
            world_points[:,:2] = np.mgrid[self.model_offset_x:self.model_width+self.model_offset_x,self.model_offset_y:self.model_height+self.model_offset_y].T.reshape(-1,2)*self.model_scale #convert to meters (scale is 1:10)
            camera_points = np.zeros((self.model_width*self.model_height,2), np.float32)

            for world_point in world_points:
                # ground_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')
                # (ground_point, jacobian) = cv2.projectPoints(ground_point, rotation_vector, translation_vector, camera_matrix, distortion_matrix)
                ground_point = np.array([[[world_point[0], world_point[1]]]], dtype='float32')
                ground_point = cv2.perspectiveTransform(ground_point, H)
                ref_point = np.array([[[world_point[0], world_point[1], -1.8*self.model_scale]]], dtype='float32')
                (ref_point, jacobian) = cv2.projectPoints(ref_point, rotation_vector, translation_vector, camera_matrix, distortion_matrix)
                # Render vertical
                im_src = cv2.line(im_src, tuple(ground_point.ravel()), tuple(ref_point.ravel()), (0,255,255), 2)

            if not cv2.imwrite('output.png',im_src):
                print("Writing failed")


        # Display images
        height, width, channel = im_src.shape
        bytesPerLine = 3 * width
        # Convert to RGB for QImage.
        cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB, im_src)
        qImg = QImage(im_src.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.viewer.setImage(QPixmap(qImg))


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    window.loadImage()
    window.loadSurface()
    sys.exit(app.exec_())
