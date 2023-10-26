import math
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import argparse
import sys
import os
import cv2
import numpy as np
import time

# vk modules
import cameras
import world_models
# import observers
import tracking
import widgets


class GraphicsScene(QtWidgets.QGraphicsScene):
    # Create signal exporting QtCore.QtCore.QPointF position.
    SceneClicked = QtCore.pyqtSignal(QtCore.QPointF)
    MouseMoved = QtCore.pyqtSignal(QtCore.QPointF)

    def __init__(self, parent=None):
        QtWidgets.QGraphicsScene.__init__(self, parent)

        self.setSceneRect(-100, -100, 200, 200)
        self.opt = False

    def set_option(self, opt):
        self.opt = opt

    def mouseMoveEvent(self, event):
        self.MouseMoved.emit(QtCore.QPointF(event.scenePos()))

    def mousePressEvent(self, event):
        # #Emit the signal
        self.SceneClicked.emit(QtCore.QPointF(event.scenePos()))


class ImageViewer(QtWidgets.QGraphicsView):
    ImageClicked = QtCore.pyqtSignal(QtCore.QPoint)
    MouseMoved = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        start = time.time()
        super(ImageViewer, self).__init__(parent)
        self.zoom = 0
        self.empty = True
        self.scene = GraphicsScene()
        self.image = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.image)
        self.setScene(self.scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.should_auto_scale = True

        # Connect the signal emitted by the GraphicsScene mousePressEvent to relay event
        self.scene.SceneClicked.connect(self.scene_clicked)
        self.scene.MouseMoved.connect(self.mouse_moved)

    def has_image(self):
        return not self.empty

    def set_cross_cursor(self, state=False):
        if state:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def fitInView(self, *__args):

        rect = QtCore.QRectF(self.image.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_image():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
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
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.image.setPixmap(pixmap)
        else:
            self.empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.image.setPixmap(QtGui.QPixmap())

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
                pass
            else:
                self.zoom = 0

    def toggle_drag_mode(self, force_no_drag=False):

        if force_no_drag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

        else:

            if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
                self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            else:
                self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mouseMoveEvent(self, event):
        super(ImageViewer, self).mouseMoveEvent(event)

    def mousePressEvent(self, event):
        self.toggle_drag_mode()
        super(ImageViewer, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.toggle_drag_mode(force_no_drag=True)
        super(ImageViewer, self).mouseReleaseEvent(event)

    def mouse_moved(self, pos):
        self.MouseMoved.emit(pos.toPoint())

    def scene_clicked(self, pos):
        # Pass local (scene) coordinates to ImageClicked()
        if self.image.isUnderMouse():
            self.ImageClicked.emit(pos.toPoint())


class MyPopup(QtWidgets.QWidget):
    def __init__(self, model):
        QtWidgets.QWidget.__init__(self)
        self.camera_model = model
        self.setWindowTitle("Correspondences")
        # Arrange layout
        popup_correspondences = QtWidgets.QVBoxLayout(self)
        self.listCorrespondences = QtWidgets.QListWidget()
        popup_correspondences.addWidget(self.listCorrespondences)

    def update_items(self):
        self.listCorrespondences.clear()

        if self.camera_model is None:
            return

        if self.camera_model.image_points.size > 0:

            print("self.camera_model.image_points", self.camera_model.image_points)
            print("self.camera_model.model_points", self.camera_model.model_points)

            # NB: model_points includes the z-axis.  Ignore that for now..
            two_d_model_points = self.camera_model.model_points[..., :2]
            assert self.camera_model.image_points.size == two_d_model_points.size

            print("two_d_model_points", two_d_model_points)

            for idx in range(0, two_d_model_points.shape[0]):
                _img_point = {"x": self.camera_model.image_points[idx][0], "y": self.camera_model.image_points[idx][1]}
                world_coordinates = self.camera_model.metric_world_point_for_image_point(image_point=_img_point)
                s = "Image (px) x:{0:.0f}, y:{1:.0f} : Surface (px) x:{2:.0f}, y:{3:.0f} : World (m) x:{4:.1f}, y:{5:.1f}".format(
                    self.camera_model.image_points[idx][0],
                    self.camera_model.image_points[idx][1],
                    two_d_model_points[idx][0],
                    two_d_model_points[idx][1],
                    world_coordinates[0],
                    world_coordinates[1]
                )

                self.listCorrespondences.addItem(s)


class Window(QtWidgets.QWidget):
    def __init__(self, debug=False):

        # TODO - clean up this interface with subclassed QGroupBox:
        # https://doc.qt.io/qt-5/qtwidgets-widgets-sliders-example.html

        super(Window, self).__init__()

        """
        Main VK Track and Image Utilities
        """
        # Each project requires a surface model, representing the world model space.
        # The world model (VKWorldModel) defines image calibration and world-to-camera-to-world translations.
        self.world_model_name = None
        self.world_model = None

        # Each project requires at least one image model.
        # The image model (VKCamera) defines the image features, image capture and seek, and various camera extrinsics.
        self.image_model = None

        # A project may deploy a player tracker.
        # The tracking objects manage the long term association of identity over frames.
        # Local (VKLocalTracker) and Global (VKGlobalTracker) trackers may be implemented.
        # VKTrackingEmulator is a special case of the VKGlobalTracker, which emulates the input
        # from a live VKGlobalTracker.
        # In most cases, a VKGlobalTracker will be implemented on the client instance, and
        # VKLocalTracker instances will be implemented on server instances.
        self.tracker = None

        """
        User interface widgets, relevant for this implementation only.
        """
        self.setWindowTitle("Camera calibration Interface")

        self.viewer = ImageViewer(self)
        self.surface = ImageViewer(self)
        self.btnLoad = QtWidgets.QToolButton(self)
        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.load_camera_image)

        self.cboSurfaces = QtWidgets.QComboBox()
        for s in ("Soccer", "Basketball", "Netball", "Hockey", "Rugby", "Tennis", "Swimming"):
            self.cboSurfaces.addItem(s)

        # Apply camera model
        self.cboSurfaces.currentIndexChanged.connect(self.set_world_model_name)

        # Compute new homography from points.
        self.btnComputeHomograhy = QtWidgets.QToolButton(self)
        self.btnComputeHomograhy.setText('Compute Homograhy')
        self.btnComputeHomograhy.clicked.connect(self.compute_homography)

        # Camera capture management
        self.btnShowCaptureDeviceWidget = QtWidgets.QToolButton(self)
        self.btnShowCaptureDeviceWidget.setText('Video Capture')
        self.btnShowCaptureDeviceWidget.clicked.connect(self.show_capture_device_controller)

        # Button to flip view
        self.btnFlipView = QtWidgets.QCheckBox(self)
        self.btnFlipView.setText('Flip View')
        self.btnFlipView.clicked.connect(self.flip_view)

        # Correspondence management
        self.btnShowCorrespondences = QtWidgets.QToolButton(self)
        self.btnShowCorrespondences.setText('Correspondences')
        self.btnShowCorrespondences.clicked.connect(self.show_correspondences)

        self.btnRemoveAllCorrespondences = QtWidgets.QToolButton(self)
        self.btnRemoveAllCorrespondences.setText('Clear All Correspondences')
        self.btnRemoveAllCorrespondences.clicked.connect(self.clear_correspondences)

        # Button to change from drag/pan to getting pixel info
        self.btnAddCorrespondences = QtWidgets.QToolButton(self)
        self.btnAddCorrespondences.setText('Add Correspondence')
        self.btnAddCorrespondences.clicked.connect(self.add_correspondences)

        # Checkable button to visualise vertical projections
        self.btnShowGridVerticals = QtWidgets.QPushButton(self)
        self.btnShowGridVerticals.setText('Vertical Projections')
        self.btnShowGridVerticals.setCheckable(True)
        self.btnShowGridVerticals.clicked.connect(self.vertical_projections)

        # Show 3d world calibration
        self.show_cal_markers = True
        self.chkShow3dCal = QtWidgets.QCheckBox(self)
        self.chkShow3dCal.setChecked(self.show_cal_markers)
        self.chkShow3dCal.setText('Show calibration Markers')
        self.chkShow3dCal.clicked.connect(self.set_cal_markers)

        # Serialise camera properties & transformation matrix
        self.btnSerialiseCameraProperties = QtWidgets.QToolButton(self)
        self.btnSerialiseCameraProperties.setText('Save Camera Properties')
        self.btnSerialiseCameraProperties.clicked.connect(self.save_camera_properties)

        # Load camera properties & transformation matrix
        self.btnLoadCameraProperties = QtWidgets.QToolButton(self)
        self.btnLoadCameraProperties.setText('Load Camera Properties')
        self.btnLoadCameraProperties.clicked.connect(self.load_camera_properties)

        # Re-centre viewpoints
        self.btnFitInView = QtWidgets.QToolButton(self)
        self.btnFitInView.setText('Re-Center Viewpoints')
        self.btnFitInView.clicked.connect(self.center_views)

        # Focal length slider
        self.sliderFocalLength = QtWidgets.QSlider(Qt.Horizontal)
        self.sliderFocalLength.setMinimum(0)
        self.sliderFocalLength.setMaximum(200)
        self.sliderFocalLength.setValue(10)
        self.sliderFocalLength.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sliderFocalLength.setTickInterval(20)
        self.sliderFocalLength.valueChanged.connect(self.update_focal_length)

        # Distortion slider
        self.sliderDistortion = QtWidgets.QSlider(Qt.Horizontal)
        self.sliderDistortion.setMinimum(0)
        self.sliderDistortion.setMaximum(30000)
        self.sliderDistortion.setValue(100)
        self.sliderDistortion.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sliderDistortion.setTickInterval(3000)
        self.sliderDistortion.valueChanged.connect(self.update_distortion_estimate)

        self.viewer.ImageClicked.connect(self.image_clicked)
        self.surface.ImageClicked.connect(self.surface_clicked)
        self.last_image_pairs = [0, 0]
        self.last_surface_pairs = [0, 0]
        self.addingCorrespondencesEnabled = False

        self.flip_image_view = False

        self.show_vertical_projections = False

        self.is_playing = False

        # Arrange layout
        vb_layout = QtWidgets.QVBoxLayout(self)
        hb_images_layout = QtWidgets.QHBoxLayout()
        hb_images_layout.addWidget(self.viewer)
        hb_images_layout.addWidget(self.surface)
        vb_layout.addLayout(hb_images_layout)

        # Distortion slider
        self.sliderVideoTime = QtWidgets.QSlider(Qt.Horizontal)
        self.sliderVideoTime.setMinimum(0)
        self.sliderVideoTime.setMaximum(0)
        self.sliderVideoTime.setValue(0)
        self.sliderVideoTime.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sliderVideoTime.setTickInterval(1)
        self.sliderVideoTime.valueChanged.connect(self.update_video_time)
        vb_layout.addWidget(self.sliderVideoTime)

        hb_layout = QtWidgets.QHBoxLayout()
        hb_layout.setAlignment(Qt.AlignLeft)
        hb_layout.addWidget(self.btnLoad)
        hb_layout.addWidget(self.btnSerialiseCameraProperties)
        hb_layout.addWidget(self.btnLoadCameraProperties)
        hb_layout.addWidget(self.cboSurfaces)
        hb_layout.addWidget(self.btnFlipView)
        hb_layout.addWidget(self.sliderFocalLength)
        hb_layout.addWidget(self.sliderDistortion)
        hb_layout.addWidget(self.btnComputeHomograhy)
        vb_layout.addLayout(hb_layout)

        hb_correspondences = QtWidgets.QHBoxLayout()
        hb_correspondences.setAlignment(Qt.AlignLeft)
        hb_correspondences.addWidget(self.btnShowCaptureDeviceWidget)
        hb_correspondences.addWidget(self.btnShowCorrespondences)
        hb_correspondences.addWidget(self.btnAddCorrespondences)
        hb_correspondences.addWidget(self.btnRemoveAllCorrespondences)
        hb_correspondences.addWidget(self.btnFitInView)
        hb_correspondences.addWidget(self.btnShowGridVerticals)
        hb_correspondences.addWidget(self.chkShow3dCal)

        vb_layout.addLayout(hb_correspondences)

        if self.world_model_name is not None:
            self.cboSurfaces.setCurrentText(self.world_model_name)

        # Widgets
        self.capture_device_control_widget = None
        self.correspondencesWidget = MyPopup(self.world_model)
        self.open_capture_devices()

    def reset_controls(self):
        # Abort correspondences
        self.last_image_pairs = {0, 0}
        self.last_surface_pairs = {0, 0}
        self.viewer.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.surface.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.btnAddCorrespondences.setStyleSheet("background-color: None")
        self.addingCorrespondencesEnabled = False
        self.viewer.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.surface.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    @staticmethod
    def scan_capture_devices():

        available_devices = []

        # Check for Vimba cameras
        vimba_cameras = cameras.enumerate_vimba_devices()
        for camera in vimba_cameras:
            # Add vimba camera object to VKCamera wrapper.
            camera_model = cameras.VKCameraVimbaDevice(device_id=camera.get_id())

            if camera_model.is_available():
                print("Found:", camera_model.__class__)
                available_devices.append(camera_model)

        return available_devices

    def update_current_camera_device(self, camera):
        # Update the image model
        self.image_model = camera

        # Apply the current flip setting
        self.flip_view()

        # Update the display
        self.view_current_frame()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.play()
            return

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_Escape:
                # Abort correspondences
                self.reset_controls()
                return

            if self.viewer.empty or self.surface.empty:
                return

    def keyReleaseEvent(self, event):
        pass

    def load_surface_image(self):

        im_src = self.world_model.surface_image()

        height, width, channel = im_src.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(im_src.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        self.surface.set_image(QtGui.QPixmap(q_img))
        self.correspondencesWidget.update_items()
        self.center_views()

    def load_camera_image(self, image_path):

        if self.world_model is not None:

            if image_path is False:
                image_path = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "/home", "Media (*.png *.xpm *.jpg *.avi *.mov *.jpg *.mp4 *.mkv *.braw)")[0]

            if os.path.splitext(image_path)[1] == ".braw":
                self.image_model = cameras.VKCameraBlackMagicRAW(filepath=image_path)
            else:
                self.image_model = cameras.VKCameraVideoFile(filepath=image_path)

            print(self.image_model)

            self.sliderVideoTime.setMaximum(max(0, self.image_model.frame_count()))

            self.view_current_frame()

    def view_current_frame(self):
        with cameras.VIMBA_INSTANCE():
            with self.image_model.vimba_camera():
                self.image_model.update_camera_properties()
                self.update_displays()
                app.processEvents()

    def set_world_model_name(self):
        print("Setting world surface model:", self.cboSurfaces.currentText())
        self.world_model_name = self.cboSurfaces.currentText()
        self.update_world_model(world_model_name=self.cboSurfaces.currentText())

    def update_world_model(self, world_model_name):
        self.world_model = world_models.VKWorldModel(sport=world_model_name)
        self.load_surface_image()
        self.center_views()

    def pix_info(self):
        # self.viewer.toggleDragMode()
        if self.addingCorrespondencesEnabled:
            self.viewer.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 100, 30)))
            self.surface.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))

    def image_clicked(self, pos):

        # Is the control key pressed?
        if self.addingCorrespondencesEnabled is True and app.queryKeyboardModifiers() == Qt.ControlModifier:
            # Draw point
            pen = QtGui.QPen(Qt.red)
            brush = QtGui.QBrush(Qt.yellow)
            self.viewer.scene.addEllipse(pos.x() - 3, pos.y() - 3, 6, 6, pen, brush)
            self.last_image_pairs = (pos.x(), pos.y())
            self.viewer.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 100, 30)))

            self.viewer.set_cross_cursor(False)
            self.surface.set_cross_cursor(True)

    def surface_clicked(self, pos):

        if self.addingCorrespondencesEnabled is True and app.queryKeyboardModifiers() == Qt.ControlModifier:
            # self.editImageCoordsInfo.setText('%d, %d' % (pos.x(), pos.y()))

            # Draw point
            pen = QtGui.QPen(Qt.red)
            brush = QtGui.QBrush(Qt.yellow)
            self.surface.scene.addEllipse(pos.x() - 3, pos.y() - 3, 6, 6, pen, brush)
            self.last_surface_pairs = (pos.x(), pos.y())  # tuple
            self.viewer.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
            self.surface.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))

            print("\n+ + + + +\nExisting image_points shape", self.world_model.image_points.shape)
            print("Existing model_points shape", self.world_model.model_points.shape)

            print("Existing image model pairs")
            print(self.world_model.image_points)
            print("Existing world model pairs")
            print(self.world_model.model_points)

            print("New image model pairs")
            print(self.last_image_pairs)
            print("New world model pairs")
            print(self.last_surface_pairs)

            self.world_model.image_points = np.append(self.world_model.image_points,
                                                      np.array([(self.last_image_pairs[0],
                                                                 self.last_image_pairs[1])], dtype='float32'), axis=0)

            self.world_model.model_points = np.append(self.world_model.model_points,
                                                      np.array([(self.last_surface_pairs[0],
                                                                 self.last_surface_pairs[1], 0)], dtype='float32'), axis=0)

            print("Updated image_points shape", self.world_model.image_points.shape)
            print("Updated model_points shape", self.world_model.model_points.shape,"\n+ + + + +")

            # Save correspondences
            self.reset_controls()

            self.viewer.set_cross_cursor(False)
            self.surface.set_cross_cursor(False)

            self.correspondencesWidget.update_items()

    def open_capture_devices(self):
        _local_devices = self.scan_capture_devices()

        if len(_local_devices) > 0:
            self.image_model = _local_devices[0]
            print("Starting up Vimba imaging device...")
            self.capture_device_control_widget = widgets.CameraControllerWidget(parent=self, devices=_local_devices)
            self.capture_device_control_widget.show()
            self.capture_device_control_widget.activateWindow()
            self.view_current_frame()
        else:
            print("No default imaging devices were found...")

    def show_capture_device_controller(self):
        if not self.capture_device_control_widget.isVisible():
            self.capture_device_control_widget.show()

        if not self.capture_device_control_widget.isActiveWindow():
            self.capture_device_control_widget.activateWindow()

    def add_correspondences(self):
        # 1. Highlight image space.
        if not self.addingCorrespondencesEnabled:
            self.addingCorrespondencesEnabled = True
            self.btnAddCorrespondences.setStyleSheet("background-color: green")
            self.viewer.set_cross_cursor(True)
            self.surface.set_cross_cursor(False)

    def show_correspondences(self):

        if not self.correspondencesWidget.isVisible():
            self.correspondencesWidget = MyPopup(self.world_model)
            self.correspondencesWidget.setGeometry(QtCore.QRect(100, 100, 600, 300))
            self.correspondencesWidget.show()

        if not self.correspondencesWidget.isActiveWindow():
            self.correspondencesWidget.activateWindow()

        self.correspondencesWidget.update_items()

    def clear_correspondences(self):
        self.correspondencesWidget.activateWindow()
        self.world_model.remove_correspondences()
        self.correspondencesWidget.update_items()
        self.update_displays()

    def compute_homography(self):
        self.world_model.compute_homography()
        self.world_model.compute_inverse_homography()
        self.image_model.surface_model = self.world_model
        self.update_displays()

    def flip_view(self):
        self.flip_image_view = self.btnFlipView.isChecked()

        if self.image_model:
            with cameras.VIMBA_INSTANCE():
                with self.image_model.vimba_camera() as cam:

                    if self.flip_image_view:
                        self.image_model.set_capture_parameters(configs={"CAP_PROP_ROTATION": cameras.VK_ROTATE_180})
                    else:
                        self.image_model.set_capture_parameters(configs={"CAP_PROP_ROTATION": cameras.VK_ROTATE_NONE})
                self.update_displays()

    def vertical_projections(self):
        self.show_vertical_projections = self.btnShowGridVerticals.isChecked()
        self.update_displays()

    def set_cal_markers(self):
        self.show_cal_markers = self.chkShow3dCal.isChecked()
        self.update_displays()

    def save_camera_properties(self):

        if self.image_model:
            path = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Camera calibration', self.cboSurfaces.currentText(), "json(*.json)")
            if path[0] != "":
                self.image_model.surface_model = self.world_model
                self.image_model.export_json(path[0])

    def load_camera_properties(self):
        """ Open a dialog for json camera file.

        Returns:
            None
        """
        config_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Camera calibration', self.cboSurfaces.currentText(), "json(*.json)")[0]
        assert os.path.exists(config_path), "Config file doesn't exist..."
        self.update_camera_properties(config_path=config_path)

        # Enable the refinement widget if a panorama image class.
        if self.image_model.__class__.__name__ == "VKCameraPanorama":
            self.stitching_control_widget = widgets.PanoramaStitcherWidget(self)
            self.stitching_control_widget.show()

    def update_camera_properties(self, config_path=None):
        """ Opens json file containing camera and model parameters and creates new VKWorldModel and VKCamera objects.

        Returns:
            None
        """

        assert os.path.exists(config_path), "Config file doesn't exist..."

        # Initialise camera and world models from file.
        self.image_model = cameras.load_camera_model(path=config_path)

        with cameras.VIMBA_INSTANCE():
            with self.image_model.vimba_camera() as cam:

                self.cboSurfaces.setCurrentText(self.image_model.surface_model.surface_model_name())
                self.world_model = self.image_model.surface_model
                self.sliderVideoTime.setMaximum(max(0, self.image_model.frame_count()))

                # Update interface
                self.correspondencesWidget.update_items()
                self.update_displays()
                self.center_views()

    def center_views(self):
        self.surface.fitInView()
        self.viewer.fitInView()

    @staticmethod
    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    def update_focal_length(self):
        # self.world_model.focal_length = self.sliderFocalLength.value()
        self.image_model.focal_length = self.sliderFocalLength.value()
        print("Updating focal length:{0}".format(self.image_model.focal_length))
        # Update the camera matrix with new focal length.
        with cameras.VIMBA_INSTANCE():
            with self.image_model.vimba_camera() as cam:
                self.image_model.update_camera_properties()

                self.update_displays()

    def update_video_time(self):
        if self.image_model is not None:
            self.image_model.set_position(frame_number=self.sliderVideoTime.value())
            self.update_displays()

    def update_distortion_estimate(self):
        self.image_model.distortion_matrix[0] = self.sliderDistortion.value() * -3e-5
        print("Updating distortion parameter: {0}".format(self.image_model.distortion_matrix[0]))
        self.update_displays()

    def play(self):

        if self.image_model is not None:
            self.is_playing = not self.is_playing
            while self.is_playing:
                self.update_displays()
                app.processEvents()
                if self.image_model.eof():
                    self.is_playing = False
        else:
            print("No image capture devices have been initialised...")

    def update_displays(self):

        if self.image_model:

            model = self.world_model
            source = self.image_model

            # Get the current image from the imaging source.
            with cameras.VIMBA_INSTANCE():
                with source.vimba_camera():
                    im_src = source.undistorted_image()

                    # Only update the surface overlay if there is an existing homography
                    if model:
                        if not model.is_homography_identity():

                            _surface_px = self.surface.image.pixmap()

                            im_out = cv2.warpPerspective(im_src,
                                                         model.homography,
                                                         (_surface_px.width(),
                                                          _surface_px.height()))

                            height, width, channel = im_out.shape
                            bytes_per_line = 3 * width
                            alpha = 0.5
                            beta = (1.0 - alpha)

                            # Composite image
                            src1 = model.surface_image()

                            if src1.shape == im_out.shape:
                                dst = cv2.addWeighted(src1, alpha, im_out, beta, 0.0)

                                # Set composite image to surface model
                                q_img = QtGui.QImage(dst.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                                self.surface.set_image(QtGui.QPixmap(q_img))

                            self.viewer.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
                            self.surface.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))

                            if self.show_cal_markers:

                                if self.show_vertical_projections:
                                    thickness = 1

                                    world_points = np.zeros((model.model_width * model.model_height, 3), np.float32)

                                    world_points[:, :2] = np.mgrid[model.model_offset_x:model.model_width + model.model_offset_x,
                                                          model.model_offset_y:model.model_height + model.model_offset_y].T.reshape(
                                        -1, 2) * model.model_scale

                                    for world_point in world_points:
                                        # Render vertical
                                        model_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')
                                        projected_ground_point = model.projected_image_point_for_3d_world_point(world_point=model_point, camera_model=source)
                                        theoretical_3d_model_point = np.array([[[world_point[0], world_point[1], -model.model_scale * 2]]], dtype='float32')
                                        projected_vertical_point = model.projected_image_point_for_3d_world_point(world_point=theoretical_3d_model_point, camera_model=source)
                                        im_src = cv2.line(im_src, tuple(projected_ground_point.ravel()), tuple(projected_vertical_point.ravel()), (0, 255, 255), thickness)
                                else:
                                    thickness = 3

                                    for world_point in model.model_points:
                                        unit_vector = -model.model_scale * 1.8

                                        # Render y-axis
                                        model_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')

                                        projected_ground_point = model.projected_image_point_for_3d_world_point(world_point=model_point, camera_model=source)
                                        theoretical_3d_model_point = np.array([[[world_point[0], world_point[1] + unit_vector, 0]]], dtype='float32')
                                        projected_vertical_point = model.projected_image_point_for_3d_world_point(world_point=theoretical_3d_model_point, camera_model=source)

                                        int_projected_ground_point = tuple(int(el) for el in tuple(projected_ground_point.ravel()))
                                        int_projected_vertical_point = tuple(int(el) for el in tuple(projected_vertical_point.ravel()))
                                        im_src = cv2.line(im_src, int_projected_ground_point, int_projected_vertical_point, (0, 255, 0), thickness)

                                        # Render x-axis
                                        model_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')
                                        projected_ground_point = model.projected_image_point_for_3d_world_point(world_point=model_point, camera_model=source)
                                        theoretical_3d_model_point = np.array([[[world_point[0] + unit_vector, world_point[1], 0]]], dtype='float32')
                                        projected_vertical_point = model.projected_image_point_for_3d_world_point(world_point=theoretical_3d_model_point, camera_model=source)

                                        int_projected_ground_point = tuple(int(el) for el in tuple(projected_ground_point.ravel()))
                                        int_projected_vertical_point = tuple(int(el) for el in tuple(projected_vertical_point.ravel()))
                                        im_src = cv2.line(im_src, int_projected_ground_point, int_projected_vertical_point, (0, 0, 255), thickness)

                                        # Render vertical
                                        model_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')
                                        projected_ground_point = model.projected_image_point_for_3d_world_point(world_point=model_point, camera_model=source)
                                        theoretical_3d_model_point = np.array([[[world_point[0], world_point[1], unit_vector]]], dtype='float32')
                                        projected_vertical_point = model.projected_image_point_for_3d_world_point(world_point=theoretical_3d_model_point, camera_model=source)

                                        int_projected_ground_point = tuple(int(el) for el in tuple(projected_ground_point.ravel()))
                                        int_projected_vertical_point = tuple(int(el) for el in tuple(projected_vertical_point.ravel()))
                                        im_src = cv2.line(im_src, int_projected_ground_point, int_projected_vertical_point, (255, 0, 0), thickness)

            # Display images
            height, width, channel = im_src.shape
            bytes_per_line = 3 * width

            # NB - Vimba devices return an image in B-R-G format.
            # PyQt does not have a specific format for BGR ordering.
            # However, we convert the image from BGR format to RGB.
            rgb_data = cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB)

            # Convert to QtGui.QImage.
            q_img = QtGui.QImage(rgb_data.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

            # Set the converted image.
            self.viewer.set_image(QtGui.QPixmap(q_img))

        else:
            print("Warning: No camera model has been initialised.")


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str, required=False,
                        help="input sport type e.g. 'Hockey'")
    parser.add_argument('--config', type=str, required=False,
                        help="input camera configuration json file.")
    parser.add_argument('--debug', type=bool, required=False,
                        help="set True to skip manual config.")

    return parser


if __name__ == '__main__':

    opts = argument_parser().parse_args(sys.argv[1:])

    app = QtWidgets.QApplication(sys.argv)
    window = Window(debug=opts.debug is not None)
    window.setGeometry(500, 300, 800, 600)

    _initial_surface_model = opts.sport or "Hockey"
    window.update_world_model(world_model_name=_initial_surface_model)

    if opts.config is not None:
        assert os.path.exists(opts.config), "Invalid path to config file."
        window.update_camera_properties(config_path=opts.config)

    if opts.debug:
        window.update_camera_properties(config_path="/data/OMB_Tests/Basketball-Panorama-Planar.json")

    window.show()
    window.center_views()

    sys.exit(app.exec_())
