"""Camera model class - handles image calibration and world-to-camera-to-world translations"""
import json
import math
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq

from world_models.helpers import *

# https://www.learnopencv.com/homography-examples-using-opencv-python-c/


class VKWorldModel:

    def __init__(self, sport):

        _sport = sport

        if _sport.__class__.__name__ == "str":
            _sport = sport_constant_with_name(sport)
        assert _sport >= 0, "WTF!! Valid sport constant not assigned..."

        print("Initialising Camera Model for {0} surface.".format(sport_name_with_constant(_sport)))
        self.sport = _sport

        # Model properties
        self.surface_properties = surface_properties_for_sport(sport=_sport)
        self.model_width = self.surface_properties["model_width"]
        self.model_height = self.surface_properties["model_height"]
        self.model_offset_x = self.surface_properties["model_offset_x"]
        self.model_offset_y = self.surface_properties["model_offset_y"]
        self.model_scale = self.surface_properties["model_scale"]

        # Image correspondences
        self.homography = np.zeros((3, 3))
        np.fill_diagonal(self.homography, 1)
        self.inverse_homography = np.zeros((3, 3))
        np.fill_diagonal(self.inverse_homography, 1)
        self.image_points = np.empty([0, 2])    # 2D coordinates system
        self.model_points = np.empty([0, 3])    # 3D coordinate system
        # print("Initial image_points shape", self.image_points.shape)
        # print("Initial model_points shape", self.model_points.shape)
        self._surface_image = None

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return "World Model for {0}:\nImage-World Homography:\n{1}".format(
            sport_name_with_constant(self.sport),
            self.homography)

    @classmethod
    def model_with_sport(cls, sport):
        """Create an instance of a camera model class.

        Returns:
            (self): God created man, in the likeness of God made He him. ...
        """
        return cls(sport=sport)

    def surface_model_name(self):
        """Surface model name for current class model.

        Returns:
            (str): Sport id as string
        """
        return sport_name_with_constant(self.sport)

    def sport(self):
        """Surface model constant for current class model.

        Returns:
            (int): Sport id as constant
        """
        return self.sport

    def surface_image(self):
        """Surface model image for current class model.

        Returns:
            (array): Returns surface image as numpy array in RGB channel order
        """
        if self._surface_image is None:
            self._surface_image = surface_image_with_sport(sport=self.sport)

        return self._surface_image

    def compute_homography(self):
        """Estimate and retain image to world homography.

        Returns:
            (array): Returns a numpy array with 3x3 matrix
        """
        self.homography, mask = cv2.findHomography(self.image_points, self.model_points)
        return self.homography

    def compute_inverse_homography(self):
        """Estimate world to image homography.

        Returns:
            (array): Returns a numpy array with 3x3 matrix
        """
        val, self.inverse_homography = cv2.invert(self.homography)
        return self.inverse_homography

    def is_homography_identity(self):
        """Evaluate homography state.

        Returns:
            (bool): Returns True if homography is identity.
        """
        identity = np.zeros((3, 3))
        np.fill_diagonal(identity, 1)
        return np.array_equal(self.homography, identity)

    def world_point_for_image_point(self, image_point):
        """Estimate world coordinates in meters (relative to top left origin) from camera coordinates.
        Args:
            image_point (x, y): camera/image coordinates.

        Returns:
            (x,y): Returns world coordinates.
        """
        world_point = cv2.perspectiveTransform(np.array([[[image_point['x'], image_point['y']]]], dtype='float32'), self.homography)
        # NB- the image-world correspondences are in pixel-to-pixel terms.
        # i.e. the world surface image correspondences are in pixel values.
        # The model scale factor translates the surface model from pixels to
        # world coordinates in meters.
        _metric_x = world_point.item(0) / self.model_scale
        _metric_y = world_point.item(1) / self.model_scale

        return _metric_x, _metric_y

    def normal_world_point_for_image_point(self, image_point):
        """Estimate normalised world coordinates [0-1] (relative to top left origin) from camera coordinates.
        Args:
            image_point (x, y): camera/image coordinates.

        Returns:
            (x,y): Returns normalised world coordinates.
        """
        world_point = self.world_point_for_image_point(image_point=image_point)
        _normal_x = world_point[0] / self.model_width
        _normal_y = world_point[1] / self.model_height

        return _normal_x, _normal_y

    def projected_image_point_for_2d_world_point(self, world_point):
        """Estimate 2d camera coordinates from 2d world coordinates.
        Args:
            world_point (x, y): world/model coordinates.

        Returns:
            (x,y): Returns world coordinates.
        """
        # TODO - the world point could be in meters or normalised coordinates.
        raise NotImplementedError('TODO- the world point could be in meters or normalised coordinates')
        projected_point = cv2.perspectiveTransform(np.array([[[world_point['x'], world_point['y']]]], dtype='float32'), self.inverse_homography)
        return projected_point.item(0), projected_point.item(1)

    def projected_image_point_for_3d_world_point(self, world_point, camera_model):
        """Estimate 2d camera coordinates from 3d world coordinates.

        Args:
            world_point (x, y, z): world/model coordinates.
            camera_model (VKCamera): pose-initialised camera with valid extrinsics.

        Returns:
            (x,y): Returns world coordinates.
        """
        assert camera_model is not None, "WTF!!  You gotta give me something..."
        # TODO - verify the camera extrinsics exist.

        # if np.all(camera_model.translation_vector == np.zeros((3, 3))):
        # TODO - only update extrinsics once.
        camera_model.estimate_camera_extrinsics(world_model=self)

        (projected_point, jacobian) = cv2.projectPoints(world_point,
                                                        camera_model.rotation_vector,
                                                        camera_model.translation_vector,
                                                        camera_model.camera_matrix,
                                                        camera_model.distortion_matrix)
        return projected_point

    def rotated_image_point_from_camera_point_with_image_point(self, image_point, camera_location, fov=10):
        """Estimates a point that is n-degrees along a plane that is orthogonal to the
        angle between the camera and the target in world coordinates.  The result is returned in
        image coordinates.

        Args:
            image_point (x,y): image-based coordinates for point of interest.
            camera_location (x,y): world-based coordinates for estimated camera location.
            fov (int): desired field of view angle from camera location.

        Returns:
            (x,y): Returns world coordinates.
        """
        # Convert image target and the camera location to world coordinates.
        _target_world_location = self.world_point_for_image_point({"x": image_point[0], "y": image_point[1]})
        _camera_world_location = self.world_point_for_image_point({"x": camera_location[0], "y": camera_location[1]})

        dy = (_camera_world_location[1] - _target_world_location[1])
        dx = (_camera_world_location[0] - _target_world_location[0])
        h = math.sqrt(pow(dx, 2) + pow(dy, 2))

        # Calculate angle from target point to camera location in radians
        camera_angle_rad = math.atan2(dy, dx)  # radians
        camera_angle_deg = camera_angle_rad * (180. / math.pi)

        # Estimate field of view
        fov_1_deg = camera_angle_deg + fov
        fov_remain_deg = 90 - fov_1_deg
        fov_1_opp_deg = 90 - fov
        fov_1_adj_deg = fov_1_opp_deg - fov_remain_deg
        fov_1_theta_deg = 90 - fov_1_adj_deg
        fov_1_adj_rad = fov_1_theta_deg * (math.pi / 180.)

        # Estimate a projection from the camera location along the field of view.
        a = h
        b = math.tan((fov * (math.pi / 180.))) * a

        # Estimate a point that fov degrees on a plane that is orthogonal to the line of sight of the camera.
        x2 = _target_world_location[0] + math.cos(fov_1_adj_rad) * b
        y2 = _target_world_location[1] - math.sin(fov_1_adj_rad) * b

        # Convert fov projection from world coordinates back to image coordinates.
        _x2, _y2 = self.projected_image_point_for_2d_world_point({"x": x2, "y": y2})

        return int(_x2), int(_y2)

    def rotated_image_crop(self, image_target, camera, fov=10):
        """Estimates a rotated image crop that mimics the perspective of an
        imaging device at a known location.

        Args:
            image_target (array): image-based coordinates for point of interest.
            camera (VKCamera): calibrated camera object.
            fov (int): desired field of view angle from camera location.

        Returns:
            (rect): Returns crop coordinates in lr, tr, bl, br order.
        """

        assert self.is_homography_identity() is False, "WTF!  [rotated_image_crop] The camera is not calibrated.."

        # Image point in camera space
        _x, _y = image_target

        # Estimate the 2d location of the camera in relation to the model bounds.
        __x, __y = camera.camera_2d_image_space_location

        # Estimate image points that are +/-fov degrees from the image target.
        # These points are along a plane that is orthogonal to the angle
        # from the camera location to the target.
        # These points are the bottom left and bottom right corners.
        bottom_left = self.rotated_image_point_from_camera_point_with_image_point(image_point=(_x, _y), camera_location=(__x, __y), fov=-fov)
        bottom_right = self.rotated_image_point_from_camera_point_with_image_point(image_point=(_x, _y), camera_location=(__x, __y), fov=fov)

        # What is the width of the line between image points?
        width = math.sqrt(pow(bottom_right[0] - bottom_left[0], 2) + pow(bottom_right[1] - bottom_left[1], 2))

        # Solve the equation for the line from the camera location to the target.
        # TODO - predict this line first, then extrapolate forwards and backwards
        # so that that target is centered in the crop.
        points = [(__x, __y), (_x, _y)]

        x_coords, y_coords = zip(*points)
        a = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(a, y_coords)[0]

        # Extrapolate along the line from the camera location to the target
        # such that the height of a crop is relative to the existing width.
        y = _y - int(width * 0.75)  # 4:3 format
        x = int((y - c) / m)

        # Using the extrapolation from the original target point, estimate
        # the top left and top right corners.
        top_left = self.rotated_image_point_from_camera_point_with_image_point(image_point=(x, y), camera_location=(__x, __y), fov=-fov)
        top_right = self.rotated_image_point_from_camera_point_with_image_point(image_point=(x, y), camera_location=(__x, __y), fov=fov)

        return top_left, top_right, bottom_left, bottom_right

    def remove_correspondences(self):
        """Reset image and world correspondences.
        """
        self.image_points = np.empty([0, 2])  # 2D coordinates system
        self.model_points = np.empty([0, 3])  # 3D coordinate system
        self.homography = np.zeros((3, 3))
        np.fill_diagonal(self.homography, 1)

