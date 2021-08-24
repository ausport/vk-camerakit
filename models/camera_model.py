import time
import json
import math
import numpy as np

from models.helpers import *


# https://www.learnopencv.com/homography-examples-using-opencv-python-c/

# TODO Compute reprojection error - mean L2 loss between 2D homography and 3D projections on the ground plane.

"""
NB - don't deal with images here - deal only with matrices and model properties.
But, provide an interface to warp images with images.
"""


class CameraModel:

    def __init__(self, sport):

        _sport = sport

        if _sport.__class__.__name__ == "str":
            _sport = sport_constant_with_name(sport)
        assert _sport >= 0, "WTF!! Valid sport constant not assigned..."

        print("Initialising Camera Model for {0} surface.".format(sport_name_with_constant(_sport)))
        self.sport = _sport

        # Model properties
        surface_properties = surface_properties_for_sport(sport=_sport)
        self.model_width = surface_properties["model_width"]
        self.model_height = surface_properties["model_height"]
        self.model_offset_x = surface_properties["model_offset_x"]
        self.model_offset_y = surface_properties["model_offset_y"]
        self.model_scale = surface_properties["model_scale"]

        # Image correspondences
        self.homography = np.zeros((3, 3))
        np.fill_diagonal(self.homography, 1)
        self.image_points = np.empty([0, 2])    # 2D coordinates system
        self.model_points = np.empty([0, 3])    # 3D coordinate system

        # Camera intrinsics properties
        self.focal_length = 0
        self.camera_matrix = np.zeros((3, 3))
        self.optimal_camera_matrix = np.zeros((3, 3))
        self.distortion_matrix = np.zeros((4, 1))

        # Camera extrinsic matrix
        # TODO - disambiguate this property with the image rotation prop in VK2.
        self.rotation_vector = None
        self.translation_vector = None
        self.camera_point = None

        self._surface_image = None

        # TODO - move all camera control stuff out of the model and into the camera controller.
        # self._source_path = None
        # self._source_image = None
        # self._video_object = None
        # self._frame_count = 0

    def __str__(self):
        return sport_name_with_constant(self.sport)

    @classmethod
    def model_with_sport(cls, sport):
        return cls(sport=sport)

    def surface_model_name(self):
        return sport_name_with_constant(self.sport)

    def sport(self):
        return self.sport

    def surface_image(self):
        if self._surface_image is None:
            self._surface_image = surface_image_with_sport(sport=self.sport)

        return self._surface_image

    # TODO - move all camera control stuff out of the model and into the camera controller.
    # def set_source_path(self, source_path):
    #
    #     _img = None
    #
    #     if os.path.exists(source_path):
    #         # Try to open the file as an image file...
    #         _img = cv2.imread(source_path)
    #         self._frame_count = 1
    #
    #         if _img is None:
    #             # Try to open as a video file
    #             self._video_object = VideoObject(video_path=source_path)
    #             image = self._video_object.get_frame(2)
    #             return image
    #             # vidcap = cv2.VideoCapture(source_path)
    #             # success, image = vidcap.read()
    #             #
    #             # if success:
    #
    #     # self._source_image = _img
    #     # self._source_path = source_path
    #     # return _img
    #     return None

    # TODO - move all camera control stuff out of the model and into the camera controller.
    # def source_path(self):
    #     return self._source_path

    def compute_homography(self):

        start = time.time()
        cv2.findHomography()
        self.homography, mask = cv2.findHomography(self.image_points, self.model_points)
        print("compute_homography(self): --> {0}".format(time.time() - start))

    def inverse_homography(self):
        start = time.time()
        if self.homography.__class__.__name__ == "NoneType":
            self.compute_homography()

        # Compute inverse of 2D homography
        val, h = cv2.invert(self.homography)
        print("inverse_homography(self): --> {0}".format(time.time() - start))
        return h

    @staticmethod
    def identity_homography():
        identity = np.zeros((3, 3))
        np.fill_diagonal(identity, 1)
        return identity

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

    def world_point_for_image_point(self, image_point):
        world_point = cv2.perspectiveTransform(np.array([[[image_point['x'], image_point['y']]]], dtype='float32'), self.homography)
        return world_point.item(0), world_point.item(1)

    def projected_image_point_for_2d_world_point(self, world_point):
        projected_point = cv2.perspectiveTransform(np.array([[[world_point['x'], world_point['y']]]], dtype='float32'), self.inverse_homography())
        return projected_point.item(0), projected_point.item(1)

    def projected_image_point_for_3d_world_point(self, world_point):

        if self.translation_vector is None:
            self.estimate_camera_extrinsics()

        (projected_point, jacobian) = cv2.projectPoints(world_point,
                                                        self.rotation_vector,
                                                        self.translation_vector,
                                                        self.camera_matrix,
                                                        self.distortion_matrix)
        return projected_point

    @staticmethod
    def ray_intersection(line1, line2):
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

    def perspective_aware_crop_for_image_point(self, image_point, fov = 5, offset = 0):

        _x, _y = image_point

        # Convert tracking location to world coordinates - using camera-specific model
        __x, __y = self.world_point_for_image_point({"x": _x, "y": _y})
        # print(_x, _y, "-->", __x, __y)

        # print("******** Compute FOV ********")
        # print("Selected Image Space Point: 	({0}, {1})".format(_x, _y))
        # print("Selected World Space Point: 	({0}, {1})".format(__x, __y))
        # print("Camera Image Space Estimate: 	{0}".format(self.camera_point))
        c = self.world_point_for_image_point({"x": self.camera_point[0], "y": self.camera_point[1]})
        # print("Camera World Space Estimate: 	{0}".format(c))
        dy = (c[1] - __y)
        dx = (c[0] - __x)
        h = math.sqrt(pow(dx, 2) + pow(dy, 2))
        # print("Projections Props (dx, dy, h): 	{0}, {1}, {2}".format(dx, dy, h))

        # Angle from user point to camera location radians
        camera_angle_rad = math.atan2(dy, dx)  # radians
        camera_angle_deg = camera_angle_rad * (180. / math.pi)

        # # Draw image space ray projection.
        # im_src = cv2.line(im_src, (int(model.camera_point[0]), int(model.camera_point[1])),
        # 				  (int(_x), int(_y)),
        # 				  (255, 255, 255), 3)

        # fov = 5
        fov_1_deg = camera_angle_deg + fov
        fov_remain_deg = 90 - fov_1_deg
        fov_1_opp_deg = 90 - fov
        fov_1_adj_deg = fov_1_opp_deg - fov_remain_deg

        fov_1_theta_deg = 90 - fov_1_adj_deg
        fov_1_adj_rad = fov_1_theta_deg * (math.pi / 180.)

        # print("Camera Angle (A): 	{0}".format(camera_angle_deg))
        # print("FOV Half (B):		{0}".format((fov / 2)))
        # print("fov_1_deg (C):		{0}".format(fov_1_deg))
        # print("fov_remain_deg (D):	{0}".format(fov_remain_deg))
        # print("fov_1_deg (E):		{0}".format(fov_1_deg))
        # print("fov_remain_deg (F):	{0}".format(fov_remain_deg))
        # print("fov_1_opp_deg (G):	{0}".format(fov_1_opp_deg))
        # print("fov_1_adj_deg (H):	{0}".format(fov_1_adj_deg))
        # print("fov_1_theta_deg (I):	{0}".format(fov_1_theta_deg))

        a = h
        b = math.tan((fov * (math.pi / 180.))) * a
        h = math.sqrt(pow(a, 2) + pow(b, 2))
        # print("Projections Right Ray (a, b, h): 	{0}, {1}, {2}".format(a, b, h))

        x2 = __x + math.cos(fov_1_adj_rad) * (b+offset)
        y2 = __y - math.sin(fov_1_adj_rad) * (b+0)

        # print("Model Ray Point dx, dy: 	({0}, {1})".format(math.cos(fov_1_adj_rad) * b, math.sin(fov_1_adj_rad) * b))
        # print("Model Ray Point 1: 	({0}, {1})".format(x2, y2))
        _x2, _y2 = self.projected_image_point_for_2d_world_point({"x": x2, "y": y2})
        # print("World Ray Point 1: 	({0}, {1})".format(_x2, _y2))

        return _x2, _y2

    # TODO - move all camera control stuff out of the model and into the camera controller.
    # def distorted_camera_image_cv2(self):
    #     return self._source_image

    # TODO - replace self._source_image with image param as input to the method.
    # We return the de-warped image but we don't retain the warped image.
    def undistorted_camera_image_cv2(self):
        if self._source_image is None:
            print("WTF- WE DON'T HAVE A SOURCE IMAGE!")
            self._source_image = np.zeros((480, 640, 3), np.uint8)

        # img = cv2.undistort(self.distorted_camera_image_cv2(),
        #                     self.camera_matrix,
        #                     self.distortion_matrix,
        #                     None,
        #                     self.optimal_camera_matrix)

        # img = cv2.fisheye.undistortImage(self.__sourceImage,
        #                        self.camera_matrix,
        #                        self.distortion_matrix)

        return cv2.undistort(self.distorted_camera_image_cv2(), self.camera_matrix, self.distortion_matrix, None, self.optimal_camera_matrix)

    # def distorted_camera_image_qimage(self):
    #     # NB But we need to convert cv2 to QImage for display in qt widgets..
    #
    #
    #     # TODO - move this to a function in a calibration helper (Qt5-specific).
    #     # The camera models should be platform and gui-independent.
    #     start = time.time()
    #     cvImg = self.distorted_camera_image_cv2()
    #     height, width, channel = cvImg.shape
    #     bytesPerLine = 3 * width
    #     cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB, cvImg)
    #     qimg =  QtGui.QImage(cvImg.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    #     print("distorted_camera_image_QtGui.QImage(...): --> {0}".format(time.time() - start))
    #
    #     return qimg

    # def undistorted_camera_image_qimage(self):
    #
    #     # TODO - move this to a function in a calibration helper (Qt5-specific).
    #     # The camera models should be platform and gui-independent.
    #
    #     cvImg = self.undistorted_camera_image_cv2()
    #     height, width, channel = cvImg.shape
    #     bytesPerLine = 3 * width
    #     cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB, cvImg)
    #     return QtGui.QImage(cvImg.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

    def update_camera_properties(self, with_distortion_matrix = None, with_camera_matrix = None, with_optimal_camera_matrix = None):

        start = time.time()
        if self._source_image is None:
            print("WTF- WE DON'T HAVE A SOURCE IMAGE!")
            self._source_image = np.zeros((480, 640, 3), np.uint8)

        if with_distortion_matrix is not None:
            self.distortion_matrix = with_distortion_matrix

        if with_camera_matrix is not None:
            self.camera_matrix = with_camera_matrix

        else:

            h, w = self._source_image.shape[:2]
            fx = 0.5 + self.focal_length / 50.0
            self.camera_matrix = np.float64([[fx * w, 0, 0.5 * (w - 1)],
                                             [0, fx * w, 0.5 * (h - 1)],
                                             [0.0, 0.0, 1.0]])

        if with_optimal_camera_matrix is not None:
            self.optimal_camera_matrix = with_optimal_camera_matrix
        else:
            self.optimal_camera_matrix = self.camera_matrix

        # print("Updating Camera Matrix:\n {0}".format(self.focal_length, self.camera_matrix))
        # print("Updating Optimal Camera Matrix:\n{0}".format(self.optimal_camera_matrix))
        # print("Updating Camera Distortion Matrix:\n{0}".format(self.distortion_matrix))
        print(self.camera_matrix)
        print("update_camera_properties(...): --> {0}".format(time.time() - start))

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
                    # 'rotation_vector': self.rotation_vector,
                    # 'translation_vector': self.translation_vector,
                    'distortion_matrix': self.distortion_matrix.tolist(),
                    'image_points': self.image_points.tolist(),
                    'model_points': self.model_points.tolist(),
                    # 'camera_point': self.camera_point,
                    'camera_matrix': self.camera_matrix.tolist()
                },
                indent=4,
                separators=(',', ': ')
            )

        print(j)

        with open(json_path[0]+".json", 'w') as data_file:
            data_file.write(j)

    # def import_camera_model(self, json_path):
    #     '''
    #     Load the camera data from the JSON file
    #     '''
    #     print("Importing", json_path[0])
    #
    #     with open(json_path[0]) as data_file:
    #         j = json.load(data_file)
    #
    #     # Verify path exists:
    #     try:
    #         image_path = j["image_path"]
    #         if os.path.isfile(image_path):
    #             vidcap = cv2.VideoCapture(image_path)
    #             success, image = vidcap.read()
    #             if success:
    #                 self.set_camera_image_from_image(image, image_path)
    #             else:
    #                 self.set_camera_image_from_file(image_path)
    #
    #     except KeyError:
    #         print(QtWidgets.QtWidgets.QApplication.topLevelWidgets()[0])
    #         image_path = QtWidgets.QFileDialog.getOpenFileName(QtWidgets.QtWidgets.QApplication.topLevelWidgets()[0], "Locate media for calibration",
    #                                                  "/home",
    #                                                  "Media (*.png *.xpm *.jpg *.avi *.mov *.jpg *.mp4 *.mkv)")
    #         if os.path.isfile(image_path[0]):
    #             vidcap = cv2.VideoCapture(image_path[0])
    #             success, image = vidcap.read()
    #
    #             if success:
    #                 self.set_camera_image_from_image(image, image_path[0])
    #             else:
    #                 self.set_camera_image_from_file(image_path[0])
    #         else:
    #             return
    #
    #     self.sport = j["surface_model"]
    #     self.set_surface_image(self.sport)
    #
    #     self.model_width = j["model_dimensions"][0]
    #     self.model_height = j["model_dimensions"][1]
    #     self.model_offset_x = j["model_offset"][0]
    #     self.model_offset_y = j["model_offset"][1]
    #     self.model_scale = j["model_scale"]
    #     self.focal_length = j["focal_length"]
    #     # self.rotation_vector = j["rotation_vector"]
    #     self.homography = np.array(j["homography"])
    #     self.distortion_matrix = np.array(j["distortion_matrix"])
    #     self.image_points = np.array(j["image_points"])
    #     self.model_points = np.array(j["model_points"])
    #     self.camera_matrix = np.array(j["camera_matrix"])
    #
    #     if "optimal_camera_matrix" in j:
    #         self.optimal_camera_matrix = np.array(j["optimal_camera_matrix"])
    #     else:
    #         self.optimal_camera_matrix = self.camera_matrix
    #
    #     self.compute_homography()
    #     self.estimate_camera_extrinsics()



