import sys, math, os
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
        self.homography, mask = cv2.findHomography(self.image_points, self.model_points)


    def inverse_homography(self):
        if self.homography.__class__.__name__ == "NoneType":
            self.compute_homography()

        # Compute inverse of 2D homography
        val, H = cv2.invert(self.homography)
        return H

    def identity_homography(self):
        return np.fill_diagonal(np.zeros((3, 3)), 1)

    def is_homography_identity(self):
        return np.array_equal(self.homography, self.identity_homography())

    def compute_camera_matrix(self):

        if self.__sourceImage is None:
            print("WTF- WE DON'T HAVE A SOURCE IMAGE!")
            self.__sourceImage = np.zeros((480, 640, 3), np.uint8)

        h, w = self.__sourceImage.shape[:2]
        fx = 0.5 + self.focal_length / 50.0
        self.camera_matrix = np.float64([[fx * w, 0, 0.5 * (w - 1)],
                                         [0, fx * w, 0.5 * (h - 1)],
                                         [0.0, 0.0, 1.0]])

        # print("Camera Matrix {0}:\n {1}".format(self.focal_length, self.camera_matrix))


    def surface_image(self):
         return QPixmap("./Surfaces/{:s}.png".format(self.sport))

    def set_surface_image(self, sport):

        self.sport = sport
        px = QPixmap("./Surfaces/{:s}.png".format(sport))
        self.surface_dimensions = px.size()
        print("Setting surface:", sport, self.surface_dimensions)
        return px

    def surface_image_cv2(self):
        return cv2.imread("./Surfaces/{:s}.png".format(self.sport))

    def set_camera_image_from_file(self, image_path):
        # NB We set the camera image as a cv2 image (numpy array).
        self.__sourceImage = cv2.imread(image_path)
        self.__image_path = image_path

    def set_camera_image_from_image(self, image, image_path):
        self.__sourceImage = image
        self.__image_path = image_path

    def distorted_camera_image_cv2(self):

        return self.__sourceImage


    def undistorted_camera_image_cv2(self):

        if self.__sourceImage is None:
            print("WTF- WE DON'T HAVE A SOURCE IMAGE!")
            self.__sourceImage = np.zeros((480, 640, 3), np.uint8)

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
        self.homography = np.zeros((3, 3))
        np.fill_diagonal(self.homography, 1)

    def reset(self):
        # Remove previous values
        self.remove_correspondences()

    def surface_properties(self, sport):
        # Return a dictionary of values for each sport.
        properties = {
            "model_width": 50,
            "model_height": 25,
            "model_offset_x": 1,
            "model_offset_y": 1,
            # Scaling factor required to convert from real world in meters to surface pixels.
            "model_scale": 10
        }

        if sport == "pool":
            return {
                "model_width": 50,
                "model_height": 25,
                "model_offset_x": 1,
                "model_offset_y": 1,
                # Scaling factor required to convert from real world in meters to surface pixels.
                "model_scale": 10
            }

        if sport == "tennis":
            return {
                "model_width": 30,
                "model_height": 15,
                "model_offset_x": 1,
                "model_offset_y": 1,
                # Scaling factor required to convert from real world in meters to surface pixels.
                "model_scale": 50
            }

        if sport == "hockey":
            return {
                "model_width": 91,
                "model_height": 55,
                "model_offset_x": 5,
                "model_offset_y": 5,
                # Scaling factor required to convert from real world in meters to surface pixels.
                "model_scale": 10
            }

        if sport == "netball":
            return {
                "model_width": 31,
                "model_height": 15,
                "model_offset_x": 3,
                "model_offset_y": 3,
                # Scaling factor required to convert from real world in meters to surface pixels.
                "model_scale": 100
            }

        return properties


    def camera_image_path(self):
        return self.__image_path

    def export_camera_model(self, json_path):
        print("Exporting", json_path[0])
        j = json.dumps(
                {
                    'surface_model': self.sport,
                    'image_path' : self.__image_path,
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

    def import_camera_model(self, json_path):
        '''
        Load the camera data from the JSON file
        '''
        print("Importing", json_path[0])

        with open(json_path[0]) as data_file:
            j = json.load(data_file)

        # Verify path exists:
        try:
            image_path = j["image_path"]
            if os.path.isfile(image_path):
                vidcap = cv2.VideoCapture(image_path)
                success, image = vidcap.read()
                if success:
                    self.set_camera_image_from_image(image, image_path)
                else:
                    self.set_camera_image_from_file(image_path)

        except KeyError:
            print(QApplication.topLevelWidgets()[0])
            image_path = QFileDialog.getOpenFileName(QApplication.topLevelWidgets()[0], "Locate media for calibration",
                                                     "/home",
                                                     "Media (*.png *.xpm *.jpg *.avi *.mov *.jpg *.mp4 *.mkv)")
            if os.path.isfile(image_path[0]):
                vidcap = cv2.VideoCapture(image_path[0])
                success, image = vidcap.read()

                if success:
                    self.set_camera_image_from_image(image, image_path[0])
                else:
                    self.set_camera_image_from_file(image_path[0])
            else:
                return

        self.sport = j["surface_model"]
        self.set_surface_image(self.sport)

        self.model_width = j["model_dimensions"][0]
        self.model_height = j["model_dimensions"][1]
        self.model_offset_x = j["model_offset"][0]
        self.model_offset_y = j["model_offset"][1]
        self.model_scale = j["model_scale"]
        self.focal_length = j["focal_length"]
        self.rotation_vector = j["rotation_vector"]
        self.homography = np.array(j["homography"])
        self.distortion_matrix = np.array(j["distortion_matrix"])
        self.image_points = np.array(j["image_points"])
        self.model_points = np.array(j["model_points"])
        self.camera_matrix = np.array(j["camera_matrix"])
        self.compute_homography()
        # self.homography = np.array(j["homography"])
        print("Imported homography:\n", self.homography)


    def __init__(self, sport="hockey"):

        self.sport = sport
        self.set_surface_image(sport)
        surface_properties = self.surface_properties(sport)
        print(surface_properties)

        # Model properties
        self.model_width = surface_properties["model_width"]
        self.model_height = surface_properties["model_height"]
        self.model_offset_x = surface_properties["model_offset_x"]
        self.model_offset_y = surface_properties["model_offset_y"]
        #Scaling factor required to convert from real world in meters to surface pixels.
        self.model_scale = surface_properties["model_scale"]

        # Camera properties
        self.homography = np.zeros((3, 3))
        np.fill_diagonal(self.homography, 1)

        self.focal_length = 0
        self.camera_matrix = None
        self.distortion_matrix = np.zeros((4, 1))
        self.rotation_vector = None
        self.translation_vector = None

        # Image correspondences
        self.image_points = np.empty([0, 2])    #2D coordinates system
        self.model_points =np.empty([0, 3])     #3D coordinate system

        self.__sourceImage = None
        self.__image_path = os.path.abspath("./Images/{:s}.png".format(sport))
        self.set_camera_image_from_file(self.__image_path)

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

        # Load camera properties & transformation matrix
        self.btnLoadCameraProperties = QToolButton(self)
        self.btnLoadCameraProperties.setText('Load Camera Properties')
        self.btnLoadCameraProperties.clicked.connect(self.load_camera_properties)

        # Focal length slider
        self.sliderFocalLength = QSlider(Qt.Horizontal)
        self.sliderFocalLength.setMinimum(0)
        self.sliderFocalLength.setMaximum(200)
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
        HBlayout.addWidget(self.btnLoadCameraProperties)
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

    def loadSurface(self, sport):
        self.surface.set_image(self.camera_model.surface_image())
        self.camera_model.set_surface_image(sport)
        self.correspondencesWidget.update_items()

    def loadImage(self):

        image_path = QFileDialog.getOpenFileName(self, "Open Image",
                                                "/home",
                                                "Media (*.png *.xpm *.jpg *.avi *.mov *.jpg *.mp4 *.mkv)")

        vidcap = cv2.VideoCapture(image_path[0])
        success, image = vidcap.read()

        if success:
            self.camera_model.set_camera_image_from_image(image, image_path[0])
        else:
            self.camera_model.set_camera_image_from_file(image_path[0])

        self.viewer.set_image(QPixmap(self.camera_model.undistorted_camera_image_qimage()))

        # Loading a new image should also negate previous data entries.
        self.camera_model.reset()
        self.loadSurface(self.cboSurfaces.currentText())


    def setCameraModel(self):

        self.camera_model = CameraModel(sport=self.cboSurfaces.currentText())
        self.loadSurface(self.cboSurfaces.currentText())

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
            # print("_mylastImagePairs:", self.last_image_pairs)
            # print("_mylastSurfacePairs:", self.last_surface_pairs)
            #
            # s = "Image x:{0}, y:{1} : Surface x:{2}, y:{3}".format(
            #         self.last_image_pairs[0],
            #         self.last_image_pairs[1],
            #         self.last_surface_pairs[0],
            #         self.last_surface_pairs[1])

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
        self.correspondencesWidget.activateWindow()
        self.camera_model.remove_correspondences()
        self.correspondencesWidget.update_items()
        self.updateDisplays()

    def save_camera_properties(self):

        if self.camera_model:
            path = QFileDialog.getSaveFileName(self, 'Save Camera Calibration', self.cboSurfaces.currentText(), "json(*.json)")
            if path[0] != "":
                self.camera_model.export_camera_model(path)

    def load_camera_properties(self):

        path = QFileDialog.getOpenFileName(self, 'Load Camera Calibration', self.cboSurfaces.currentText(), "json(*.json)")
        if path[0] != "":
            self.camera_model.import_camera_model(path)
            # self.cboSurfaces.setCurrentText(self.camera_model.sport)
            self.updateDisplays()
            self.correspondencesWidget.update_items()

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

            #Update homography
            model.compute_camera_matrix()
            model.compute_homography()

            # Get model sample image
            im_src = model.undistorted_camera_image_cv2()

            # Estimate naive camera intrinsics (camera matrix)
            camera_matrix = model.camera_matrix

            # Distortion matrix
            distortion_matrix = model.distortion_matrix

            # Warp source image to destination based on homography
            print(model.surface_dimensions)
            print(model.homography)

            # Only update the surface overlay if there is an existing homography
            if not model.is_homography_identity():

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
                height, width, channel = im_out.shape
                bytesPerLine = 3 * width
                alpha = 0.5
                beta = (1.0 - alpha)

                # Composite image
                cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB, im_out)
                src1 = model.surface_image_cv2()
                dst = cv2.addWeighted(src1, alpha, im_out, beta, 0.0)

                # Set composite image to surface model
                qImg = QImage(dst.data, width, height, bytesPerLine, QImage.Format_RGB888)
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
    window.loadSurface("tennis")
    sys.exit(app.exec_())
