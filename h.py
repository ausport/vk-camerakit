import sys, math
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json
# https://www.learnopencv.com/homography-examples-using-opencv-python-c/

# TODO Compute reprojection error - mean L2 loss between 2D homography and 3D projections on the ground plane.


class CameraModel:

    def compute_homography(self):
        self.homography, status = cv2.findHomography(self.image_points, self.model_points)
        # print("Image Homograhy :\n {0}".format(self.homography))


    def inverse_homography(self):
        if self.homography.__class__.__name__ == "NoneType":
            self.compute_homography()

        # Compute inverse of 2D homography
        val, H = cv2.invert(self.homography)
        return H

    def compute_camera_matrix(self):
        h, w = self.__sourceImage.shape[:2]
        fx = 0.5 + self.focal_length / 50.0
        self.camera_matrix = np.float64([[fx * w, 0, 0.5 * (w - 1)],
                                         [0, fx * w, 0.5 * (h - 1)],
                                         [0.0, 0.0, 1.0]])

        # print("Camera Matrix {0}:\n {1}".format(self.focal_length, self.camera_matrix))


    def surface_image(self):
        
        px = QPixmap("./Surfaces/{:s}.png".format(self.sport))
        self.surface_dimensions = px.size()
        print("Loading surface:", self.sport, self.surface_dimensions)
        return px


    def set_camera_image(self, image_path):
        # NB We set the camera image as a cv2 image (numpy array).
        self.__sourceImage = cv2.imread(image_path)


    def distorted_camera_image_cv2(self):

        return self.__sourceImage


    def undistorted_camera_image_cv2(self):

        img = cv2.undistort(self.__sourceImage,
                               self.camera_matrix,
                               self.distortion_matrix)
        return img

    def distorted_camera_image_qimage(self):
        # NB But we need to convert cv2 to QImage for display in qt widgets..

        cvImg = self.distorted_camera_image_cv2()
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB, cvImg)
        return QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)

    def undistorted_camera_image_qimage(self):

        cvImg = self.undistorted_camera_image_cv2()
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB, cvImg)
        return QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)


    def remove_correspondences(self):

        self.image_points = np.empty([0, 2])  # 2D coordinates system
        self.model_points = np.empty([0, 3])  # 3D coordinate system

    def reset(self):
        # Remove previous values
        self.remove_correspondences()


    def export_camera_model(self, json_path):
        print("Exporting", json_path[0])
        j = json.dumps(
                {
                    'surface_model': self.sport,
                    'model_dimensions': [self.model_width, self.model_height],
                    'model_offset': [self.model_offset_x, self.model_offset_y],
                    'model_scale': self.model_scale,
                    'homography': self.homography.tolist(),
                    'focal_length': self.focal_length,
                    'rotation_vector': self.rotation_vector,
                    # 'translation_vector': self.translation_vector,
                    'distortion_matrix': self.distortion_matrix.tolist(),
                    'image_points': self.image_points.tolist(),
                    'model_points': self.model_points.tolist(),
                    'camera_matrix': self.camera_matrix.tolist()
                },
                indent=4,
                separators=(',', ': ')
            )

        print(j)

        with open(json_path[0]+".json", 'w') as data_file:
            data_file.write(j)

    def import_camera_model(self):
        '''
        Load the camera data from the JSON file
        '''
        pass
        # self.space_transform = SpaceTransform(video_path=self.video_path)
        # self.pool_length = self.space_transform.pool_length
        # self.inv_transform_matrix = np.linalg.inv(self.space_transform.transform_matrix)
        # top, bottom = 0, 1
        # left, right = 0, 1
        # # Inverse transform from work space back to screen space
        # self.points = np.array([[[left, top],  # pylint: disable=C0326
        #                          [left, bottom],
        #                          [right, bottom],
        #                          [right, top]]],  # pylint: disable=C0326
        #                        dtype=np.float32)
        # self.points = cv2.perspectiveTransform(self.points,
        #                                        self.inv_transform_matrix)[0]

    def __bool__(self):
        return self.__bool__

    def __init__(self, sport="hockey"):

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
        self.image_points = np.empty([0, 2])    #2D coordinates system
        self.model_points =np.empty([0, 3])     #3D coordinate system

        self.__sourceImage = None
        self.set_camera_image("./Images/{:s}.png".format(sport))

        #Internal validation
        self.__bool__ = False

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
            self.__bool__ = True

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

            self.distortion_matrix[0] = -0.17751
            print ("Distortion Matrix :\n {0}".format(self.distortion_matrix))

            self.focal_length = 7
            print("Focal Length :\n {0}".format(self.focal_length))
            self.__bool__ = True

        elif sport == "hockey":
            # Tennis
            # Distorted
            self.image_points = np.array([(630, 104), (920, 193), (108, 225), (52, 121)], dtype='float32')
            # Undistorted
            # self.image_points = np.array([(964, 162), (964, 600), (508, 600), (964, 490)], dtype='float32')
            self.model_points = np.array([(964, 600, 0), (508, 600, 0), (508, 162, 0), (964, 162, 0)],
                                         dtype='float32')
            self.model_width = 91
            self.model_height = 55
            self.model_offset_x = 5
            self.model_offset_y = 5
            # Scaling factor required to convert from real world in meters to surface pixels.
            self.model_scale = 10

            self.distortion_matrix[0] = -0.023880000000000002
            print("Distortion Matrix :\n {0}".format(self.distortion_matrix))

            self.focal_length = 80
            print("Focal Length :\n {0}".format(self.focal_length))
            self.__bool__ = True

        elif sport == "netball":
            # Tennis
            # Distorted
            self.image_points = np.array([(122, 1143), (1241, 1056), (3810, 1360), (3436, 1751)], dtype='float32')
            # Undistorted
            # self.image_points = np.array([(964, 162), (964, 600), (508, 600), (964, 490)], dtype='float32')
            self.model_points = np.array([(308, 1827, 0), (308, 308, 0), (3352, 308, 0), (3352, 1827, 0)],
                                         dtype='float32')
            self.model_width = 31
            self.model_height = 15
            self.model_offset_x = 3
            self.model_offset_y = 3
            # Scaling factor required to convert from real world in meters to surface pixels.
            self.model_scale = 100

            self.distortion_matrix[0] = 0.
            print("Distortion Matrix :\n {0}".format(self.distortion_matrix))

            self.focal_length = 21
            print("Focal Length :\n {0}".format(self.focal_length))
            self.__bool__ = True


        if self.__bool__:
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

    def set_cross_cursor(self, state = False):
        if state:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

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

    def toggleDragMode(self, forceNoDrag = False):

        if forceNoDrag:
            self.setDragMode(QGraphicsView.NoDrag)

        else:

            if self.dragMode() == QGraphicsView.ScrollHandDrag:
                self.setDragMode(QGraphicsView.NoDrag)
            else:
                self.setDragMode(QGraphicsView.ScrollHandDrag)

    # def toggleCrossCursor(self):
    #     if self.cursor() == QGraphicsView.CrossCursor:
    #         self.setDragMode(QGraphicsView.NoDrag)
    #     else:
    #         self.setDragMode(QGraphicsView.CrossCursor)

    def mousePressEvent(self, event):
        # TODO - only execute this if the space bar is pressed to indicate adding a coorresponence point.
        # if event.key() == Qt.Key_Space:
        #   super(ImageViewer, self).mousePressEvent(event)
        self.toggleDragMode()
        super(ImageViewer, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.toggleDragMode(forceNoDrag=True)
        super(ImageViewer, self).mouseReleaseEvent(event)

    def scene_clicked(self, pos):
        # Pass local (scene) coordinates to ImageClicked()
        print("scene_clicked")
        if self.image.isUnderMouse():
            self.ImageClicked.emit(pos.toPoint())


class MyPopup(QWidget):
    def __init__(self, model):
        QWidget.__init__(self)
        self.camera_model = model
        self.setWindowTitle("Correspondences")
        # Arrange layout
        popup_Correspondences = QVBoxLayout(self)
        self.listCorrespondences = QListWidget()
        popup_Correspondences.addWidget(self.listCorrespondences)

    def update_items(self):
        self.listCorrespondences.clear()

        if self.camera_model.image_points.size > 0:

            print("self.camera_model.image_points", self.camera_model.image_points)
            print("self.camera_model.model_points", self.camera_model.model_points)

            #NB: model_points includes the z-axis.  Ignore that for now..
            two_d_model_points = self.camera_model.model_points[...,:2]
            assert self.camera_model.image_points.size == two_d_model_points.size

            print("two_d_model_points", two_d_model_points)

            for idx in range(0, two_d_model_points.shape[0]):
                print(idx)
                s = "Image x:{0}, y:{1} : Surface x:{2}, y:{3}".format(
                    self.camera_model.image_points[idx][0],
                    self.camera_model.image_points[idx][1],
                    two_d_model_points[idx][0],
                    two_d_model_points[idx][1])

                self.listCorrespondences.addItem(s)


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Camera Calibration Interface")

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

        # Compute new homography from points.
        self.btnComputeHomograhy = QToolButton(self)
        self.btnComputeHomograhy.setText('Compute Homograhy')
        self.btnComputeHomograhy.clicked.connect(self.updateDisplays)

        # Correspondence management
        self.btnShowCorrespondences = QToolButton(self)
        self.btnShowCorrespondences.setText('Show Correspondences')
        self.btnShowCorrespondences.clicked.connect(self.showCorrespondences)

        self.btnRemoveAllCorrespondences = QToolButton(self)
        self.btnRemoveAllCorrespondences.setText('Clear All Correspondences')
        self.btnRemoveAllCorrespondences.clicked.connect(self.clearCorrespondences)
        
        # Button to change from drag/pan to getting pixel info
        self.btnAddCorrespondences = QToolButton(self)
        self.btnAddCorrespondences.setText('Add Correspondence')
        self.btnAddCorrespondences.clicked.connect(self.addCorrespondences)

        # Serialise camera properties & transformation matrix
        self.btnSerialiseCameraProperties = QToolButton(self)
        self.btnSerialiseCameraProperties.setText('Save Camera Properties')
        self.btnSerialiseCameraProperties.clicked.connect(self.save_camera_properties)


        self.editImageCoordsInfo = QLineEdit(self)
        self.editImageCoordsInfo.setReadOnly(True)
        # Focal length slider
        self.sliderFocalLength = QSlider(Qt.Horizontal)
        self.sliderFocalLength.setMinimum(0)
        self.sliderFocalLength.setMaximum(80)
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

        self.viewer.ImageClicked.connect(self.ImageClicked)
        self.surface.ImageClicked.connect(self.SurfaceClicked)
        self.last_image_pairs = {0, 0}
        self.last_surface_pairs = {0, 0}
        self.addingCorrespondencesEnabled = False

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
        HBlayout.addWidget(self.btnSerialiseCameraProperties)
        HBlayout.addWidget(self.editImageCoordsInfo)
        HBlayout.addWidget(self.cboSurfaces)
        HBlayout.addWidget(self.sliderFocalLength)
        HBlayout.addWidget(self.sliderDistortion)
        HBlayout.addWidget(self.btnComputeHomograhy)
        VBlayout.addLayout(HBlayout)

        HB_Correspondences = QHBoxLayout()
        HB_Correspondences.setAlignment(Qt.AlignLeft)
        HB_Correspondences.addWidget(self.btnShowCorrespondences)
        HB_Correspondences.addWidget(self.btnAddCorrespondences)
        HB_Correspondences.addWidget(self.btnRemoveAllCorrespondences)

        VBlayout.addLayout(HB_Correspondences)

        self.correspondencesWidget = MyPopup(self.camera_model)


    def reset_controls(self):
        # Abort corresponances
        self.last_image_pairs = {0, 0}
        self.last_surface_pairs = {0, 0}
        self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.btnAddCorrespondences.setStyleSheet("background-color: None")
        self.addingCorrespondencesEnabled = False
        self.viewer.setDragMode(QGraphicsView.NoDrag)
        self.surface.setDragMode(QGraphicsView.NoDrag)

    # def mousePressEvent(self, event):
    #     print("Windows Mouse Event")
    #     # return event

    def keyPressEvent(self, event):
        # print("down")
        if not event.isAutoRepeat():
            if event.key() == Qt.Key_Escape:
                # Abort corresponances
                self.reset_controls()
                return

            if self.viewer.empty or self.surface.empty:
                return

        # else:
        # if event.key() == Qt.Key_Space:
        #     self.viewer.set_cross_cursor(True)
        #     self.surface.set_cross_cursor(True)
            #
            # self.viewer.setCursor(Qt.CrossCursor)
            # self.surface.setCursor(Qt.CrossCursor)


    def keyReleaseEvent(self, event):
        pass
        # if event.key() == Qt.Key_Space:
        #     self.viewer.set_cross_cursor(False)
        #     self.surface.set_cross_cursor(False)

        # if not event.isAutoRepeat():
        #     if event.key() == Qt.Key_Space:
        #         self.viewer.toggleDragMode()
        #         self.surface.toggleDragMode()

    def loadSurface(self):
        self.surface.set_image(self.camera_model.surface_image())
        self.correspondencesWidget.update_items()

    def loadImage(self):

        image_path = QFileDialog.getOpenFileName(self, "Open Image",
                                                "/home",
                                                "Images (*.png *.xpm *.jpg)")

        self.camera_model.set_camera_image(image_path[0])
        self.viewer.set_image(QPixmap(self.camera_model.undistorted_camera_image_qimage()))

        # Loading a new image should also negate previous data entries.
        self.camera_model.reset()
        self.loadSurface()


    def setCameraModel(self):

        self.camera_model = CameraModel(sport=self.cboSurfaces.currentText())
        self.loadSurface()

    def pixInfo(self):
        # self.viewer.toggleDragMode()
        if self.addingCorrespondencesEnabled:
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 100, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

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

    def ImageClicked(self, pos):

        print("ImageClicked")

        #Is the control key pressed?
        if self.addingCorrespondencesEnabled == True and app.queryKeyboardModifiers() == Qt.ControlModifier:
            # self.editImageCoordsInfo.setText('%d, %d' % (pos.x(), pos.y()))
            print("Image Points:", pos.x(), pos.y())
            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.viewer.scene.addEllipse(pos.x() - 3, pos.y() - 3, 6, 6, pen, brush)
            # self.viewer.toggleDragMode()
            self.last_image_pairs = (pos.x(), pos.y())
            # self.editModelCoords.setStyleSheet("background-color: rgb(0, 255, 0);")
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 100, 30)))

            self.viewer.set_cross_cursor(False)
            self.surface.set_cross_cursor(True)




    def SurfaceClicked(self, pos):
        print("SurfaceClicked", pos)
        if self.addingCorrespondencesEnabled == True and app.queryKeyboardModifiers() == Qt.ControlModifier:
            # self.editImageCoordsInfo.setText('%d, %d' % (pos.x(), pos.y()))

            #Draw point
            pen = QPen(Qt.red)
            brush = QBrush(Qt.yellow)
            self.surface.scene.addEllipse(pos.x() - 3, pos.y() - 3, 6, 6, pen, brush)
            # self.surface.toggleDragMode()
            self.last_surface_pairs = (pos.x(), pos.y())    #tuple
            # self.editModelCoords.setStyleSheet("background-color: rgb(0, 255, 0);")
            self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            print("_mylastImagePairs:", self.last_image_pairs)
            print("_mylastSurfacePairs:", self.last_surface_pairs)

            s = "Image x:{0}, y:{1} : Surface x:{2}, y:{3}".format(
                    self.last_image_pairs[0],
                    self.last_image_pairs[1],
                    self.last_surface_pairs[0],
                    self.last_surface_pairs[1])

            print("## EXISTING PAIRS ##")
            print(self.camera_model.image_points)
            print(self.camera_model.model_points)
            print(self.camera_model.model_points.shape)

            print("## LAST PAIRS ##")
            print(self.last_surface_pairs)
            # print(self.last_surface_pairs.shape)

            self.camera_model.image_points = np.append(self.camera_model.image_points,
                                                       np.array([(self.last_image_pairs[0],
                                                                  self.last_image_pairs[1])], dtype='float32'), axis=0)

            self.camera_model.model_points = np.append(self.camera_model.model_points,
                                                       np.array([(self.last_surface_pairs[0],
                                                                  self.last_surface_pairs[1], 0)], dtype='float32'), axis=0)

            self.editImageCoordsInfo.setText(s)

            #Save correspondences
            self.reset_controls()

            self.viewer.set_cross_cursor(False)
            self.surface.set_cross_cursor(False)

            self.correspondencesWidget.update_items()

    def addCorrespondences(self):
        #1. Highlight image space.
        if not self.addingCorrespondencesEnabled:
            self.addingCorrespondencesEnabled = True
            self.btnAddCorrespondences.setStyleSheet("background-color: green")
            self.viewer.set_cross_cursor(True)
            self.surface.set_cross_cursor(False)

    def showCorrespondences(self):

        if not self.correspondencesWidget.isVisible():
            self.correspondencesWidget = MyPopup(self.camera_model)
            self.correspondencesWidget.setGeometry(QRect(100, 100, 400, 200))
            self.correspondencesWidget.show()

        if not self.correspondencesWidget.isActiveWindow():
            self.correspondencesWidget.activateWindow()

        self.correspondencesWidget.update_items()

    def clearCorrespondences(self):
        self.w.activateWindow()

    def save_camera_properties(self):

        if self.camera_model:
            path = QFileDialog.getSaveFileName(self, 'Save File', self.cboSurfaces.currentText(), "json(*.json)")
            if path[0] != "":
                self.camera_model.export_camera_model(path)


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

        if self.camera_model:

            model = self.camera_model

            # Get model sample image
            im_src = model.undistorted_camera_image_cv2()

            # Estimate naive camera intrinsics (camera matrix)
            camera_matrix = model.camera_matrix

            # Distortion matrix
            distortion_matrix = model.distortion_matrix

            # Warp source image to destination based on homography
            im_out = cv2.warpPerspective(im_src,
                                         model.homography,
                                         (model.surface_dimensions.width(),
                                          model.surface_dimensions.height()))

            if model.image_points.size > 0:
                # Render image coordinate boundaries.
                cv2.line(im_src, (int(model.image_points[0][0]), int(model.image_points[0][1])),
                         (int(model.image_points[1][0]), int(model.image_points[1][1])), (255, 255, 0), 1)
                cv2.line(im_src, (int(model.image_points[2][0]), int(model.image_points[2][1])),
                         (int(model.image_points[1][0]), int(model.image_points[1][1])), (255, 0, 255), 1)
                cv2.line(im_src, (int(model.image_points[2][0]), int(model.image_points[2][1])),
                         (int(model.image_points[3][0]), int(model.image_points[3][1])), (0, 255, 0), 1)
                cv2.line(im_src, (int(model.image_points[0][0]), int(model.image_points[0][1])),
                         (int(model.image_points[3][0]), int(model.image_points[3][1])), (0, 255, 255), 1)

            if model.model_points.size > 0:
                # Render surface coordinate boundaries.
                cv2.line(im_out, (int(model.model_points[0][0]), int(model.model_points[0][1])),
                         (int(model.model_points[1][0]), int(model.model_points[1][1])), (255, 255, 0), 2)
                cv2.line(im_out, (int(model.model_points[2][0]), int(model.model_points[2][1])),
                         (int(model.model_points[1][0]), int(model.model_points[1][1])), (255, 0, 255), 2)
                cv2.line(im_out, (int(model.model_points[2][0]), int(model.model_points[2][1])),
                         (int(model.model_points[3][0]), int(model.model_points[3][1])), (0, 255, 0), 2)
                cv2.line(im_out, (int(model.model_points[0][0]), int(model.model_points[0][1])),
                         (int(model.model_points[3][0]), int(model.model_points[3][1])), (0, 255, 255), 2)

            # Display undistored images.
            # TODO alpha composite warped image on background surface.
            height, width, channel = im_out.shape
            bytesPerLine = 3 * width
            cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB, im_out)
            qImg = QImage(im_out.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.surface.set_image(QPixmap(qImg))

            self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

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

                world_points = np.zeros((model.model_width*model.model_height,3), np.float32)
                world_points[:,:2] = np.mgrid[model.model_offset_x:model.model_width+model.model_offset_x,model.model_offset_y:model.model_height+model.model_offset_y].T.reshape(-1,2)*model.model_scale #convert to meters (scale is 1:10)
                # camera_points = np.zeros((model.model_width*model.model_height,2), np.float32)

                for world_point in model.model_points:
                    ground_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')
                    (ground_point, jacobian) = cv2.projectPoints(ground_point, rotation_vector, translation_vector, camera_matrix, distortion_matrix)
                    # ground_point = np.array([[[world_point[0], world_point[1]]]], dtype='float32')
                    # ground_point = cv2.perspectiveTransform(ground_point, inverse_homography)
                    ref_point = np.array([[[world_point[0], world_point[1], -model.model_scale]]], dtype='float32')
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

                self.sliderFocalLength.setValue(int(model.focal_length))
                self.sliderDistortion.setValue(model.distortion_matrix[0] / -3e-5)
        else:
            print("Warning: No camera model has been initialised.")
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    window.loadSurface()
    sys.exit(app.exec_())
