import sys, math
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
# https://www.learnopencv.com/homography-examples-using-opencv-python-c/

class CameraModel:

    def compute_homography(self):
        self.homography, status = cv2.findHomography(self.image_points, self.model_points)
        print("Image Homograhy :\n {0}".format(self.homography))


    def inverse_homography(self):
        if self.homography.__class__.__name__ == "NoneType":
            self.compute_homography()

        # Compute inverse of 2D homography
        val, H = cv2.invert(self.homography)
        return H

    def compute_camera_matrix(self):
        h, w = self.sourceImage.shape[:2]
        fx = 0.5 + self.focal_length / 50.0
        self.camera_matrix = np.float64([[fx * w, 0, 0.5 * (w - 1)],
                                         [0, fx * w, 0.5 * (h - 1)],
                                         [0.0, 0.0, 1.0]])

        print("Camera Matrix {0}:\n {1}".format(self.focal_length, self.camera_matrix))


    def surfaceImage(self):
        
        px = QPixmap("./Surfaces/{:s}.png".format(self.sport))
        self.surface_dimensions = px.size()
        print("Loading surface:", self.sport, self.surface_dimensions)
        return px


    def __init__(self, sport="tennis"):

        self.sport = sport
        
        # Model properties
        self.model_width = 50
        self.model_height = 25
        self.model_offset_x = 1
        self.model_offset_y = 1
        #Scaling factor required to convert from real world in meters to surface pixels.
        self.model_scale = 10

        # Camera properties
        self.homography = None
        self.focal_length = 10.8
        self.camera_matrix = None
        self.distortion_matrix = np.zeros((4, 1))
        self.rotation_vector = None
        self.translation_vector = None
        self.surface_dimensions = None

        # Image correspondences
        self.image_points = np.array([])
        self.model_points = np.array([])

        if sport == "pool":
            # Pool
            self.image_points = np.array([(832, 889), (155, 1394), (3046, 887),(3695, 1412)], dtype='float32')
            self.model_points = np.array([(10, 10, 0), (10, 260, 0), (510, 10, 0), (510, 260, 0)], dtype='float32')
            self.model_width = 50
            self.model_height = 25
            self.model_offset_x = 1
            self.model_offset_y = 1
            #Scaling factor required to convert from real world in meters to surface pixels.
            self.model_scale = 10


        elif sport == "tennis":
            # Tennis
            # Distorted
            self.image_points = np.array([(67, 293), (484, 288), (353, 157),(230, 158)], dtype='float32')
            # Undistorted
            self.image_points = np.array([(37, 299), (490, 290), (353, 157), (228, 156)], dtype='float32')
            self.model_points = np.array([(157, 102, 0), (157, 580, 0), (1343, 580, 0), (1343, 102, 0)], dtype='float32')
            self.model_width = 30
            self.model_height = 15
            self.model_offset_x = 1
            self.model_offset_y = 1
            # Scaling factor required to convert from real world in meters to surface pixels.
            self.model_scale = 50

            self.distortion_matrix[0] = -0.26055
            print ("Distortion Matrix :\n {0}".format(self.distortion_matrix))

            self.sourceImage = cv2.imread("./Images/{:s}.png".format(sport))
            print("Image dimensions :\n {0}".format(self.sourceImage.shape))


        # Compute the camera matrix, including focal length and distortion.
        self.compute_camera_matrix()
        # Compute the homography with the camera matrix, image points and surface points.
        self.compute_homography()


class GraphicsScene(QGraphicsScene):
    # Create signal exporting QPointF position.
    SceneClicked = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

        self.setSceneRect(-100, -100, 200, 200)
        self.opt = ""

    def set_option(self, opt):
        self.opt = opt

    def mousePressEvent(self, event):
        # #Emit the signal
        self.SceneClicked.emit(QPointF(event.scenePos()))


class ImageViewer(QGraphicsView):
    ImageClicked = pyqtSignal(QPoint)

    def __init__(self, parent):
        super(ImageViewer, self).__init__(parent)
        self.zoom = 0
        self.empty = True
        self.scene = GraphicsScene()
        self.image = QGraphicsPixmapItem()
        self.scene.addItem(self.image)
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)

        # Connect the signal emitted by the GraphicsScene mousePressEvent to relay event
        self.scene.SceneClicked.connect(self.scene_clicked)

    def has_image(self):
        return not self.empty

    def fitInView(self, *__args):

        rect = QRectF(self.image.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_image():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self.zoom = 0

    def set_image(self, pixmap=None):
        self.zoom = 0
        if pixmap and not pixmap.isNull():
            self.empty = False
            self.setDragMode(QGraphicsView.NoDrag)
            self.image.setPixmap(pixmap)
        else:
            self.empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self.image.setPixmap(QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.has_image():
            if event.angleDelta().y() > 0:
                factor = 1.1
                self.zoom += 1
            else:
                factor = 0.9
                self.zoom -= 1

            if self.zoom > 0:
                self.scale(factor, factor)
            elif self.zoom == 0:
                self.fitInView()
            else:
                self.zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        super(ImageViewer, self).mousePressEvent(event)

    def scene_clicked(self, pos):
        # Pass local (scene) coordinates to ImageClicked()
        if self.image.isUnderMouse():
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

        # Apply camera model
        self.cboSurfaces.currentIndexChanged.connect(self.setCameraModel)

        # Button to change from drag/pan to getting pixel info
        self.btnAddCorrespondances = QToolButton(self)
        self.btnAddCorrespondances.setText('Add Correspondance')
        self.btnAddCorrespondances.clicked.connect(self.addCorrespondances)
        self.btnComputeHomograhy = QToolButton(self)
        self.btnComputeHomograhy.setText('Compute Homograhy')
        self.btnComputeHomograhy.clicked.connect(self.updateDisplays)
        self.editImageCoordsInfo = QLineEdit(self)
        self.editImageCoordsInfo.setReadOnly(True)
        # Focal length slider
        self.sliderFocalLength = QSlider(Qt.Horizontal)
        self.sliderFocalLength.setMinimum(0)
        self.sliderFocalLength.setMaximum(50)
        self.sliderFocalLength.setValue(10)
        self.sliderFocalLength.setTickPosition(QSlider.TicksBelow)
        self.sliderFocalLength.setTickInterval(1)
        self.sliderFocalLength.valueChanged.connect(self.updateFocalLength)
        # Distortion slider
        self.sliderDistortion = QSlider(Qt.Horizontal)
        self.sliderDistortion.setMinimum(0)
        self.sliderDistortion.setMaximum(30000)
        self.sliderDistortion.setValue(100)
        self.sliderDistortion.setTickPosition(QSlider.TicksBelow)
        self.sliderDistortion.setTickInterval(1)
        self.sliderDistortion.valueChanged.connect(self.updateDistortionEstimate)

        self.listCorrespondances = QListWidget()
        self.viewer.ImageClicked.connect(self.ImageClicked)
        self.surface.ImageClicked.connect(self.SurfaceClicked)
        self.last_image_pairs = {0, 0}
        self.last_surface_pairs = {0, 0}
        self.addingCorrespondancesEnabled = False
  
        self.camera_model = CameraModel(self.cboSurfaces.currentText())

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
        HBlayout.addWidget(self.sliderDistortion)
        HBlayout.addWidget(self.btnComputeHomograhy)
        VBlayout.addLayout(HBlayout)

    def reset_controls(self):
        # Abort corresponances
        self.last_image_pairs = {0, 0}
        self.last_surface_pairs = {0, 0}
        self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.btnAddCorrespondances.setStyleSheet("background-color: None")
        self.addingCorrespondancesEnabled = False
        self.viewer.setDragMode(QGraphicsView.NoDrag)
        self.surface.setDragMode(QGraphicsView.NoDrag)

    def keyPressEvent(self, event):

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_Escape:
                # Abort corresponances
                self.reset_controls()
                return

            if self.viewer.empty or self.surface.empty:
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
        self.surface.set_image(self.camera_model.surfaceImage())

    def loadImage(self):
        self.viewer.set_image(QPixmap("./Images/{:s}.png".format(self.cboSurfaces.currentText())))

    def setCameraModel(self):

        self.loadImage()
        self.loadSurface()

    def pixInfo(self):
        # self.viewer.toggleDragMode()
        if self.addingCorrespondancesEnabled:
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 100, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

    def ImageClicked(self, pos):
        print("Image Points:", pos.x(), pos.y())

        if self.viewer.dragMode()  == QGraphicsView.NoDrag and self.addingCorrespondancesEnabled == True:
            # self.editImageCoordsInfo.setText('%d, %d' % (pos.x(), pos.y()))
            print(pos.x(), pos.y())
            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.viewer.scene.addEllipse(pos.x() - 3, pos.y() - 3, 6, 6, pen, brush)
            self.viewer.toggleDragMode()
            self.last_image_pairs = {pos.x(), pos.y()}
            # self.editModelCoords.setStyleSheet("background-color: rgb(0, 255, 0);")
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 100, 30)))
            print("_mylastImagePairs:", self.last_image_pairs)


    def draw_image_space_detection(self, pos):

        # Render reference point annotation.
        r = 5
        yellow = Qt.yellow
        pen = QPen(Qt.red)
        brush = QBrush(QColor(255, 255, 0, 100))

        poly = QPolygonF()
        x, y = pos.x(), pos.y()
        poly_points = np.array([])
#         #
#         # # Compute inverse of 2D homography
#         # print("**", homography)
#         #
#         val, H = cv2.invert(self.homography)
#         #
#         for i in range(1, 33):
#             # These points are in world coordinates.
#             _x = x + (r * math.cos(2 * math.pi * i / 32))
#             _y = y + (r * math.sin(2 * math.pi * i / 32))
#
#                 # ground_point = np.array([[[world_point[0], world_point[1]]]], dtype='float32')
#                 # ground_point = cv2.perspectiveTransform(ground_point, H)
#                 # ref_point = np.array([[[world_point[0], world_point[1], -10]]], dtype='float32')
#                 # (ref_point, jacobian) = cv2.projectPoints(ref_point, rotation_vector, translation_vector, camera_matrix, distortion_matrix)
#                 # # Render vertical
#                 # im_src = cv2.line(im_src, tuple(ground_point.ravel()), tuple(ref_point.ravel()), (0,255,255), 2)
#
#
#             #Convert to image coordinates.
#             axis = np.float32([[_x, _y]]).reshape(-1,2)
#             imgpts = cv2.perspectiveTransform(axis, H)
#
#             #Draw the points in a circle in perspective.
#             (xx, yy) = tuple(imgpts[0].ravel())
#
#             poly_points = np.append(poly_points, [xx, yy])
#
#             _p = QPointF(xx,yy)
#             poly.append(QPointF(xx,yy))
#
#         self.viewer.scene.addPolygon(poly, pen, brush)
#
#         #Render image-space point
#         axis = np.float32([[pos.x(),pos.y(),0], [pos.x(),pos.y(),-20]]).reshape(-1,3)
#         (imgpts, jacobian) = cv2.projectPoints(axis,
#                                                self._myRotationVector,
#                                                self._myTranslationVector,
#                                                self._myCameraMatrix,
#                                                self._myDistortionMatrix)
#
#         (x, y) = tuple(imgpts[0].ravel())
#         (xx, yy) = tuple(imgpts[1].ravel())
#         self.viewer.scene.addLine(xx, yy, x, y, pen)


    def SurfaceClicked(self, pos):

        if not self.camera_model.homography.__class__.__name__ == "NoneType":
            print("ok")
            print("Surface Points:", pos.x(), pos.y())
            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.surface.scene.addEllipse(pos.x() - 3, pos.y() - 3, 6, 6, pen, brush)
            self.surface.toggleDragMode()
            self.last_surface_pairs = {pos.x(), pos.y()}

            self.draw_image_space_detection(pos)


        if self.surface.dragMode()  == QGraphicsView.NoDrag and self.addingCorrespondancesEnabled == True:
            # self.editImageCoordsInfo.setText('%d, %d' % (pos.x(), pos.y()))
            print(pos.x(), pos.y())
            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.surface.scene.addEllipse(pos.x() - 3, pos.y() - 3, 6, 6, pen, brush)
            self.surface.toggleDragMode()
            self.last_surface_pairs = {pos.x(), pos.y()}
            # self.editModelCoords.setStyleSheet("background-color: rgb(0, 255, 0);")
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            print("_mylastSurfacePairs:", self.last_surface_pairs)

            self.editImageCoordsInfo.setText(
                "Image x:{0}, y:{1} : Surface x:{2}, y:{3}".format(
                    list(self.last_image_pairs)[0],
                    list(self.last_image_pairs)[1],
                    list(self.last_surface_pairs)[0],
                    list(self.last_surface_pairs)[1]))

            #Save corresponances
            self.reset_controls()

    def addCorrespondances(self):
        #1. Highlight image space.
        if not self.addingCorrespondancesEnabled:
            self.addingCorrespondancesEnabled = True
            self.btnAddCorrespondances.setStyleSheet("background-color: green")

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img


    def updateFocalLength(self):
        self.camera_model.focal_length = self.sliderFocalLength.value()
        print("Updating focal length:{0}".format(self.camera_model.focal_length ))
        # Update the camera matrix with new focal length.
        self.camera_model.compute_camera_matrix()

        self.updateDisplays()

    def updateDistortionEstimate(self):
        self.camera_model.distortion_matrix[0] = self.sliderDistortion.value() * -3e-5
        print("Updating distortion parameter:{0}".format(self.camera_model.distortion_matrix[0]))
        self.updateDisplays()

    def updateDisplays(self):

        model = self.camera_model

        # Get model sample image
        im_src = model.undistorted_image()

        # Estimate naive camera intrinsics (camera matrix)
        camera_matrix = model.camera_matrix
        # Distortion matrix
        distortion_matrix = model.distortion_matrix

        #Undistort image
        im_src = cv2.undistort(im_src,
                               camera_matrix,
                               distortion_matrix)

        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(im_src,
                                     model.homography,
                                     (model.surface_dimensions.width(),
                                      model.surface_dimensions.height()))

        # Display images in QT
        height, width, channel = im_out.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB, im_out)
        qImg = QImage(im_out.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.surface.set_image(QPixmap(qImg))

        self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

        # Render pool boundaries.
        cv2.line(im_src, ( int(model.image_points[0][0]), int(model.image_points[0][1])), ( int(model.image_points[1][0]), int(model.image_points[1][1])), (255,255,0), 1)
        cv2.line(im_src, ( int(model.image_points[2][0]), int(model.image_points[2][1])), ( int(model.image_points[1][0]), int(model.image_points[1][1])), (255,0,255), 1)
        cv2.line(im_src, ( int(model.image_points[2][0]), int(model.image_points[2][1])), ( int(model.image_points[3][0]), int(model.image_points[3][1])), (0,255,0), 1)
        cv2.line(im_src, ( int(model.image_points[0][0]), int(model.image_points[0][1])), ( int(model.image_points[3][0]), int(model.image_points[3][1])), (0,255,255), 1)

        # NB Generate square calibration corresponances using existing homography.
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
            world_points = np.zeros((model.model_width*model.model_height,3), np.float32)
            world_points[:,:2] = np.mgrid[model.model_offset_x:model.model_width+model.model_offset_x,model.model_offset_y:model.model_height+model.model_offset_y].T.reshape(-1,2)*model.model_scale #convert to meters (scale is 1:10)
            camera_points = np.zeros((model.model_width*model.model_height,2), np.float32)
        else:
            # Manually set world point calibration for local extrinsics.

            # Top left of pool
            world_points = np.float32([[10,10,0], [60,10,0], [10,60,0],  [60,60,0]])
            camera_points = np.zeros((2*2,2), np.float32)

        # Estimate image point coordinates on the ground plane for world point calibration markers.
        i = 0
        inverse_homography = model.inverse_homography()
        
        for world_point in world_points:
            # Remove z-axis (assumed to be zero)
            point = np.array([[[world_point[0], world_point[1]]]], dtype='float32')
            # Get image coordinates for world coordinate point (x,y).
            camera_points[i] = cv2.perspectiveTransform(point, inverse_homography)
            i = i + 1

        # Solve rotation and translation matrices
        (_, rotation_vector, translation_vector) = cv2.solvePnP(world_points, camera_points, camera_matrix, distortion_matrix)
        # print ("Rotation Matrix :\n {0}".format(rotation_vector))
        # print ("Translation Matrix :\n {0}".format(translation_vector))

        # Top left of pool space
        axis = np.float32([[60,10,0], [10,60,0], [10,10,-50]]).reshape(-1,3)
        # Bottom right of pool space
        # axis = np.float32([[500,250,0], [510,240,0], [510,250,-50]]).reshape(-1,3)

        # Render reference point annotation.
        (imgpts, jacobian) = cv2.projectPoints(axis,
                                               rotation_vector,
                                               translation_vector,
                                               camera_matrix,
                                               distortion_matrix)

        im_src = self.draw(im_src, camera_points, imgpts)

        if False:
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

            world_points = np.zeros((model.model_width*model.model_height,3), np.float32)
            world_points[:,:2] = np.mgrid[model.model_offset_x:model.model_width+model.model_offset_x,model.model_offset_y:model.model_height+model.model_offset_y].T.reshape(-1,2)*model.model_scale #convert to meters (scale is 1:10)
            # camera_points = np.zeros((model.model_width*model.model_height,2), np.float32)

            for world_point in world_points:
                # ground_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')
                # (ground_point, jacobian) = cv2.projectPoints(ground_point, rotation_vector, translation_vector, camera_matrix, distortion_matrix)
                ground_point = np.array([[[world_point[0], world_point[1]]]], dtype='float32')
                ground_point = cv2.perspectiveTransform(ground_point, inverse_homography)
                ref_point = np.array([[[world_point[0], world_point[1], -1.8*model.model_scale]]], dtype='float32')
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
        self.viewer.set_image(QPixmap(qImg))

        self.sliderFocalLength.setValue(model.focal_length)
        self.sliderDistortion.setValue(model.distortion_matrix[0] / -3e-5)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    window.loadImage()
    window.loadSurface()
    sys.exit(app.exec_())
