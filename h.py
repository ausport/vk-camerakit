import sys, math, os
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
# from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json
import time

# https://www.learnopencv.com/homography-examples-using-opencv-python-c/

# TODO Compute reprojection error - mean L2 loss between 2D homography and 3D projections on the ground plane.


class CameraModel:

	def ray_intersection(self, line1, line2):
		xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
		ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

		def det(a, b):
			return a[0] * b[1] - a[1] * b[0]

		div = det(xdiff, ydiff)
		if div == 0:
			raise Exception('rays do not intersect')

		d = (det(*line1), det(*line2))
		x = det(d, xdiff) / div
		y = det(d, ydiff) / div
		return x, y

	def compute_homography(self):

		start = time.time()
		self.homography, mask = cv2.findHomography(self.image_points, self.model_points)
		print("compute_homography(self): --> {0}".format(time.time() - start))

	def inverse_homography(self):
		start = time.time()
		if self.homography.__class__.__name__ == "NoneType":
			self.compute_homography()

		# Compute inverse of 2D homography
		val, H = cv2.invert(self.homography)
		print("inverse_homography(self): --> {0}".format(time.time() - start))
		return H

	def identity_homography(self):
		return np.fill_diagonal(np.zeros((3, 3)), 1)

	def is_homography_identity(self):
		return np.array_equal(self.homography, self.identity_homography())

	def estimate_camera_extrinsics(self):

		world_points = self.model_points
		camera_points = self.image_points
		(_, rotation_vector, translation_vector) = cv2.solvePnP(world_points,
																camera_points,
																self.camera_matrix,
																self.distortion_matrix)

		self.rotation_vector = rotation_vector
		self.translation_vector = translation_vector

		return _, rotation_vector, translation_vector, self.estimate_camera_point()

	def estimate_camera_point(self):

		assert len(self.model_points >= 2), "Not enough model points to estimate camera point"

		world_points = self.model_points

		pt1 = np.array([[[world_points[0][0], world_points[0][1], 0]]], dtype='float32')
		pt2 = np.array([[[world_points[0][0], world_points[0][1], -self.model_scale]]], dtype='float32')

		(pt1_projection, jacobian) = cv2.projectPoints(pt1,
													   self.rotation_vector,
													   self.translation_vector,
													   self.camera_matrix,
													   self.distortion_matrix)

		(pt2_projection, jacobian) = cv2.projectPoints(pt2,
													   self.rotation_vector,
													   self.translation_vector,
													   self.camera_matrix,
													   self.distortion_matrix)

		line1 = [[pt1_projection[0][0][0], pt1_projection[0][0][1]],
				 [pt2_projection[0][0][0], pt2_projection[0][0][1]]]

		print("******* estimate_camera_point *******\n", pt1, pt2, line1)

		pt1 = np.array([[[world_points[-1][0], world_points[-1][1], 0]]], dtype='float32')
		pt2 = np.array([[[world_points[-1][0], world_points[-1][1], -self.model_scale]]], dtype='float32')

		(pt1_projection, jacobian) = cv2.projectPoints(pt1,
													   self.rotation_vector,
													   self.translation_vector,
													   self.camera_matrix,
													   self.distortion_matrix)

		(pt2_projection, jacobian) = cv2.projectPoints(pt2,
													   self.rotation_vector,
													   self.translation_vector,
													   self.camera_matrix,
													   self.distortion_matrix)

		line2 = [[pt1_projection[0][0][0], pt1_projection[0][0][1]],
				 [pt2_projection[0][0][0], pt2_projection[0][0][1]]]

		print("*******\n", pt1, pt2, line2)

		self.camera_point = self.ray_intersection(line1, line2)
		print("Camera Point ==> {0}\n*******".format(self.camera_point))
		return self.camera_point

	def projected_image_point_for_3d_world_point(self, world_point):

		if self.translation_vector is None:
			self.estimate_camera_extrinsics()

		(projected_point, jacobian) = cv2.projectPoints(world_point,
														self.rotation_vector,
														self.translation_vector,
														self.camera_matrix,
														self.distortion_matrix)
		return projected_point

	def update_camera_properties(self, with_distortion_matrix = None, with_camera_matrix = None, with_optimal_camera_matrix = None):

		start = time.time()
		if self.__sourceImage is None:
			print("WTF- WE DON'T HAVE A SOURCE IMAGE!")
			self.__sourceImage = np.zeros((480, 640, 3), np.uint8)

		if not with_distortion_matrix is None:
			self.distortion_matrix = with_distortion_matrix

		if not with_camera_matrix is None:
			self.camera_matrix = with_camera_matrix

		else:

			h, w = self.__sourceImage.shape[:2]
			fx = 0.5 + self.focal_length / 50.0
			self.camera_matrix = np.float64([[fx * w, 0, 0.5 * (w - 1)],
											 [0, fx * w, 0.5 * (h - 1)],
											 [0.0, 0.0, 1.0]])

		if not with_optimal_camera_matrix is None:
			self.optimal_camera_matrix = with_optimal_camera_matrix
		else:
			self.optimal_camera_matrix = self.camera_matrix

		# print("Updating Camera Matrix:\n {0}".format(self.focal_length, self.camera_matrix))
		# print("Updating Optimal Camera Matrix:\n{0}".format(self.optimal_camera_matrix))
		# print("Updating Camera Distortion Matrix:\n{0}".format(self.distortion_matrix))
		print(self.camera_matrix)
		print("update_camera_properties(...): --> {0}".format(time.time() - start))

	def surface_image(self):
		 return QPixmap("./Surfaces/{:s}.png".format(self.sport))

	def set_surface_image(self, sport):

		start = time.time()
		self.sport = sport
		px = QPixmap("./Surfaces/{:s}.png".format(sport))
		self.surface_dimensions = px.size()
		# print("Setting surface:", sport, self.surface_dimensions)
		print("set_surface_image(...): --> {0}".format(time.time() - start))

		return px

	def surface_image_cv2(self):
		start = time.time()
		img = cv2.imread("./Surfaces/{:s}.png".format(self.sport))
		print("surface_image_cv2(...): --> {0}".format(time.time() - start))
		return img

	def set_camera_image_from_file(self, image_path):
		# NB We set the camera image as a cv2 image (numpy array).
		start = time.time()
		self.__sourceImage = cv2.imread(image_path)
		self.__image_path = image_path
		print("set_camera_image_from_file(...): --> {0}".format(time.time() - start))

	def set_camera_image_from_image(self, image, image_path):
		self.__sourceImage = image
		self.__image_path = image_path

	def distorted_camera_image_cv2(self):

		return self.__sourceImage


	def undistorted_camera_image_cv2(self):

		start = time.time()
		if self.__sourceImage is None:
			print("WTF- WE DON'T HAVE A SOURCE IMAGE!")
			self.__sourceImage = np.zeros((480, 640, 3), np.uint8)

		# img = cv2.undistort(self.distorted_camera_image_cv2(),
		#                     self.camera_matrix,
		#                     self.distortion_matrix,
		#                     None,
		#                     self.optimal_camera_matrix)

		# img = cv2.fisheye.undistortImage(self.__sourceImage,
		#                        self.camera_matrix,
		#                        self.distortion_matrix)

		return cv2.undistort(self.distorted_camera_image_cv2(), self.camera_matrix, self.distortion_matrix, None, self.optimal_camera_matrix)

	def distorted_camera_image_qimage(self):
		# NB But we need to convert cv2 to QImage for display in qt widgets..

		start = time.time()
		cvImg = self.distorted_camera_image_cv2()
		height, width, channel = cvImg.shape
		bytesPerLine = 3 * width
		cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB, cvImg)
		qimg =  QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
		print("distorted_camera_image_qimage(...): --> {0}".format(time.time() - start))

		return qimg

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
		self.focal_length = 0
		self.camera_matrix = np.zeros((3, 3))
		self.optimal_camera_matrix = np.zeros((3, 3))
		self.distortion_matrix = np.zeros((4, 1))
		self.rotation_vector = None
		self.translation_vector = None

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
					'translation_vector': self.translation_vector,
					'distortion_matrix': self.distortion_matrix.tolist(),
					'image_points': self.image_points.tolist(),
					'model_points': self.model_points.tolist(),
					'camera_point': self.camera_point,
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

		if "optimal_camera_matrix" in j:
			self.optimal_camera_matrix = np.array(j["optimal_camera_matrix"])
		else:
			self.optimal_camera_matrix = self.camera_matrix

		self.compute_homography()
		self.estimate_camera_extrinsics()

	def __init__(self, sport="hockey"):

		_start = time.time()
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
		self.camera_matrix = np.zeros((3, 3))
		self.optimal_camera_matrix = np.zeros((3, 3))
		self.distortion_matrix = np.zeros((4, 1))
		self.rotation_vector = None
		self.translation_vector = None

		# Image correspondences
		self.image_points = np.empty([0, 2])    #2D coordinates system
		self.model_points = np.empty([0, 3])     #3D coordinate system

		# Camera extrinsics
		# TODO - disambiguate this property with the image rotation prop in VK2.
		self.rotation_vector = None
		self.translation_vector = None
		self.camera_point = None

		self.__sourceImage = None
		self.__image_path = os.path.abspath("./Images/{:s}.png".format(sport))
		self.set_camera_image_from_file(self.__image_path)

		# Compute the camera matrix, including focal length and distortion.
		self.update_camera_properties()

		# Compute the homography with the camera matrix, image points and surface points.
		self.compute_homography()
		print("self.compute_homography() --> {0}".format(time.time() - start))
		print("init_main(...): --> {0}".format(time.time() - _start))
 
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
		start = time.time()
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

		print("init_ImageViewer(...): --> {0}".format(time.time() - start))

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
		start = time.time()
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

		# Checkable button to visualise vertical projections
		self.btnShowGridVerticals = QPushButton(self)
		self.btnShowGridVerticals.setText('Vertical Projections')
		self.btnShowGridVerticals.setCheckable(True)
		self.btnShowGridVerticals.clicked.connect(self.vertical_projections)

		# Switch to OMB mode
		self.btnOMBmode = QPushButton(self)
		self.btnOMBmode.setText('OMB')
		self.btnOMBmode.setCheckable(True)
		self.btnOMBmode.clicked.connect(self.enable_OMB)

		# Show 3d world calibration
		self.show_cal_markers = True
		self.chkShow3dCal = QCheckBox(self)
		self.chkShow3dCal.setChecked(self.show_cal_markers)
		self.chkShow3dCal.setText('Show Calibration Markers')
		self.chkShow3dCal.clicked.connect(self.set_cal_markers)

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

		self.show_vertical_projections = False
		self.OMB_mode = False

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
		HB_Correspondences.addWidget(self.btnShowGridVerticals)
		HB_Correspondences.addWidget(self.btnOMBmode)
		HB_Correspondences.addWidget(self.chkShow3dCal)

		VBlayout.addLayout(HB_Correspondences)

		self.correspondencesWidget = MyPopup(self.camera_model)

		if False:
			self.camera_model.set_camera_image_from_file("/Users/stuartmorgan/Dropbox/_py/qtpy/left01.jpg")
			self.viewer.set_image(QPixmap(self.camera_model.undistorted_camera_image_qimage()))

		print("Window(QWidget): --> {0}".format(time.time() - start))

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

		start = time.time()
		self.surface.set_image(self.camera_model.surface_image())
		self.camera_model.set_surface_image(sport)
		self.correspondencesWidget.update_items()
		print("loadSurface(self, sport): --> {0}".format(time.time() - start))

	def loadImage(self):

		image_path = QFileDialog.getOpenFileName(self, "Open Image",
												"/home",
												"Media (*.png *.xpm *.jpg *.avi *.mov *.jpg *.mp4 *.mkv)")

		vidcap = cv2.VideoCapture(image_path[0])
		success, image = vidcap.read()

		if success:
			self.camera_model.set_camera_image_from_image(image, image_path[0])
			print("Loaded image: {0}".format(image_path[0]))
		else:
			self.camera_model.set_camera_image_from_file(image_path[0])

		self.viewer.set_image(QPixmap(self.camera_model.undistorted_camera_image_qimage()))
		self.updateDisplays()


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

		print("{0}".format(pos))






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

	def vertical_projections(self):
		self.show_vertical_projections = self.btnShowGridVerticals.isChecked()
		self.updateDisplays()

	def enable_OMB(self):
		self.OMB_mode = self.btnOMBmode.isChecked()
		self.updateDisplays()

	def set_cal_markers(self):
		self.show_cal_markers = self.chkShow3dCal.isChecked()
		self.updateDisplays()

	def doCheckerboardCalibration(self):

		import numpy as np
		import cv2
		import glob

		if self.camera_model:

			model = self.camera_model

			CHECKERBOARD = (9, 7)

			# termination criteria
			subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

			objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
			objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

			# Arrays to store object points and image points from all the images.
			objpoints = []  # 3d point in real world space
			imgpoints = []  # 2d points in image plane.

			img = model.distorted_camera_image_cv2()
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# Find the chess board corners
			ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

			# If found, add object points, image points (after refining them)
			if ret:
				objpoints.append(objp)
				corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
				imgpoints.append(corners)
				cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

			else:
				return

			height, width, channel = img.shape
			img_size = (img.shape[1], img.shape[0])

			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
			newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

			# Update the camera properties with updated camera intrinsics.
			model.update_camera_properties(dist, mtx, newcameramtx)

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
		self.camera_model.update_camera_properties()

		self.updateDisplays()

	def updateDistortionEstimate(self):
		self.camera_model.distortion_matrix[0] = self.sliderDistortion.value() * -3e-5
		print("Updating distortion parameter: {0}".format(self.camera_model.distortion_matrix[0]))
		self.updateDisplays()

	def updateDisplays(self, crop=None):

		if self.camera_model:

			model = self.camera_model

			#Update homography
			start = time.time()
			model.compute_homography()
			print("model.compute_homography() --> {0}".format(time.time() - start))

			# Get model sample image
			start = time.time()
			im_src = model.undistorted_camera_image_cv2()
			print("model.undistorted_camera_image_cv2() --> {0}".format(time.time() - start))

			# Estimate naive camera intrinsics (camera matrix)
			camera_matrix = model.camera_matrix

			# Distortion matrix
			distortion_matrix = model.distortion_matrix

			# Only update the surface overlay if there is an existing homography
			if not model.is_homography_identity():

				start = time.time()
				im_out = cv2.warpPerspective(im_src,
											 model.homography,
											 (model.surface_dimensions.width(),
											  model.surface_dimensions.height()))
				print("cv2.warpPerspective() --> {0}".format(time.time() - start))

				start = time.time()

				print("render points --> {0}".format(time.time() - start))

				# Display undistored images.
				height, width, channel = im_out.shape
				bytesPerLine = 3 * width
				alpha = 0.5
				beta = (1.0 - alpha)

				# Composite image
				start = time.time()
				cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB, im_out)
				src1 = model.surface_image_cv2()
				dst = cv2.addWeighted(src1, alpha, im_out, beta, 0.0)
				print("cv2.addWeighted() --> {0}".format(time.time() - start))

				# Set composite image to surface model
				qImg = QImage(dst.data, width, height, bytesPerLine, QImage.Format_RGB888)
				self.surface.set_image(QPixmap(qImg))

				self.viewer.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
				self.surface.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

				# Solve camera extrinsics - rotation and translation matrices
				start = time.time()
				# _, rotation_vector, translation_vector, camera_point = model.estimate_camera_extrinsics()

				print("cv2.solvePnP() --> {0}".format(time.time() - start))

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
							projected_ground_point = model.projected_image_point_for_3d_world_point(model_point)
							theoretical_3d_model_point = np.array([[[world_point[0], world_point[1], -model.model_scale*2]]], dtype='float32')
							projected_vertical_point = model.projected_image_point_for_3d_world_point(theoretical_3d_model_point)
							im_src = cv2.line(im_src, tuple(projected_ground_point.ravel()), tuple(projected_vertical_point.ravel()), (0,255,255), thickness)
					else:
						thickness = 3

						for world_point in model.model_points:
							unit_vector = -model.model_scale * 1.8

							# Render y-axis
							model_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')
							projected_ground_point = model.projected_image_point_for_3d_world_point(model_point)
							theoretical_3d_model_point = np.array([[[world_point[0], world_point[1]+unit_vector, 0]]], dtype='float32')
							projected_vertical_point = model.projected_image_point_for_3d_world_point(theoretical_3d_model_point)
							im_src = cv2.line(im_src, tuple(projected_ground_point.ravel()), tuple(projected_vertical_point.ravel()), (0, 255, 0), thickness)

							# Render x-axis
							model_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')
							projected_ground_point = model.projected_image_point_for_3d_world_point(model_point)
							theoretical_3d_model_point = np.array([[[world_point[0]+unit_vector, world_point[1], 0]]], dtype='float32')
							projected_vertical_point = model.projected_image_point_for_3d_world_point(theoretical_3d_model_point)
							im_src = cv2.line(im_src, tuple(projected_ground_point.ravel()), tuple(projected_vertical_point.ravel()), (0, 0, 255), thickness)

							# Render vertical
							model_point = np.array([[[world_point[0], world_point[1], 0]]], dtype='float32')
							projected_ground_point = model.projected_image_point_for_3d_world_point(model_point)
							theoretical_3d_model_point = np.array([[[world_point[0], world_point[1], unit_vector]]], dtype='float32')
							projected_vertical_point = model.projected_image_point_for_3d_world_point(theoretical_3d_model_point)
							im_src = cv2.line(im_src, tuple(projected_ground_point.ravel()), tuple(projected_vertical_point.ravel()), (255, 0, 0), thickness)

					# if not cv2.imwrite('output.png',im_src):
					#     print("Writing failed")

			# Display images
			height, width, channel = im_src.shape
			bytesPerLine = 3 * width

			# Convert to RGB for QImage.
			cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB, im_src)
			qImg = QImage(im_src.data, width, height, bytesPerLine, QImage.Format_RGB888)

			self.viewer.set_image(QPixmap(qImg))

			# self.sliderFocalLength.setValue(int(model.focal_length))
			# self.sliderDistortion.setValue(model.distortion_matrix[0] / -3e-5)
		else:
			print("Warning: No camera model has been initialised.")



if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	start = time.time()
	window = Window()
	print("window = Window(): --> {0}".format(time.time() - start))

	window.setGeometry(500, 300, 800, 600)

	start = time.time()
	window.show()
	print("window.show(): --> {0}".format(time.time() - start))

	start = time.time()
	window.loadSurface("hockey")
	print("window.loadSurface(): --> {0}".format(time.time() - start))

	sys.exit(app.exec_())
