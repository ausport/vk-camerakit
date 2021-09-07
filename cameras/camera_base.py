"""Generic class for all image sources"""
import cv2
import filetype
import numpy as np
from PIL import Image

from models.geometry import ray_intersection


class VKCamera:
    def __init__(self, verbose_mode=False):
        """Constructor for generic image source class.
        Use subclassed instances of this class

        Args:
            verbose_mode (bool): Additional class detail logging.
        """

        self.verbose_mode = verbose_mode
        self.video_object = None

        # Camera intrinsics properties
        self.focal_length = 23.
        self.camera_matrix = np.zeros((3, 3))
        self.optimal_camera_matrix = np.zeros((3, 3))
        self.distortion_matrix = np.zeros((4, 1))

        # Camera extrinsic properties
        # TODO - disambiguate this property with the image rotation prop in VK2.
        self.rotation_vector = None
        self.translation_vector = None
        self.camera_2d_image_space_location = None

        if verbose_mode:
            print(self)

    def get_frame(self):
        """Takes the next available frame from the relevant camera object.
         This method MUST be overridden by child classes
        """
        raise NotImplementedError

    def frame_count(self):
        """The number of frames in the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_COUNT property - zero if a live camera.
        """
        return int(self.video_object.get(cv2.CAP_PROP_FRAME_COUNT))

    def frame_position(self):
        """The current frame number in the video resource.

        Returns:
            (int): The CAP_PROP_POS_FRAMES property - zero if a live camera.
        """
        return int(self.video_object.get(cv2.CAP_PROP_POS_FRAMES))

    def width(self):
        """The pixel width of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_WIDTH property.
        """
        return int(self.video_object.get(cv2.CAP_PROP_FRAME_WIDTH))

    def height(self):
        """The pixel height of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_HEIGHT property.
        """
        return int(self.video_object.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def fps(self):
        """The frames per second of the video resource.

        Returns:
            (int): The CAP_PROP_FPS property.
        """
        return int(self.video_object.get(cv2.CAP_PROP_FPS))

    def eof(self):
        """Signals end of video file.

        Returns:
            (bool): True is end of file.
        """
        return int(self.video_object.get(cv2.CAP_PROP_POS_FRAMES)) >= int(self.video_object.get(cv2.CAP_PROP_FRAME_COUNT))

    def undistorted_image(self, image=None):
        """Undistorted input image with camera matrices.

        Args:
            image (array): (optional) input image.

        Returns:
            (array): Distortion-corrected image.
        """
        image = image or self.get_frame()
        return cv2.undistort(image, self.camera_matrix, self.distortion_matrix, None, self.optimal_camera_matrix)

    def update_camera_properties(self, with_distortion_matrix=None, with_camera_matrix=None, with_optimal_camera_matrix=None):
        """Recalculates the camera intrinsics matrix.

        Args:
            with_distortion_matrix (array): (optional) 4x1 numpy array defining camera distortion.
            with_camera_matrix (array): (optional) 3x3 numpy array defining camera dimension matrix.
            with_optimal_camera_matrix (array): (optional) 3x3 numpy array overriding default camera dimension matrix.
        """
        if with_distortion_matrix is not None:
            self.distortion_matrix = with_distortion_matrix

        if with_camera_matrix is not None:
            self.camera_matrix = with_camera_matrix

        else:

            h, w = self.height(), self.width()
            fx = 0.5 + self.focal_length / 50.0
            self.camera_matrix = np.float64([[fx * w, 0, 0.5 * (w - 1)],
                                             [0, fx * w, 0.5 * (h - 1)],
                                             [0.0, 0.0, 1.0]])

        if with_optimal_camera_matrix is not None:
            self.optimal_camera_matrix = with_optimal_camera_matrix
        else:
            self.optimal_camera_matrix = self.camera_matrix

    def estimate_camera_extrinsics(self, world_model):
        """Estimates the rotation and the translation vectors for the camera using the geometric properties of a calibrated world surface.

        Args:
            world_model (VKWorldModel): calibrated world surface model with camera/image coordinates.

        Returns:
            (int): Success.
            (array): 3x1 rotation vector.
            (array): 3x1 translation vector.
            (x,y): predicted camera location in 2d image space coordinates.
        """
        h, w = self.height(), self.width()
        fx = 0.5 + self.focal_length / 50.0
        # TODO - figure out the interpreter warning.
        self.camera_matrix = np.float64([[fx * w, 0., 0.5 * (w - 1.)], [0., fx * w, 0.5 * (h - 1.)], [0., 0., 1.]])

        (_, rotation_vector, translation_vector) = cv2.solvePnP(world_model.model_points,
                                                                world_model.image_points,
                                                                self.camera_matrix,
                                                                self.distortion_matrix)

        self.rotation_vector = rotation_vector
        self.translation_vector = translation_vector
        self.camera_2d_image_space_location = self.estimate_camera_location_with_world(world_model=world_model)

        return _, self.rotation_vector, self.translation_vector, self.camera_2d_image_space_location

    def estimate_camera_location_with_world(self, world_model):
        """Estimate a 2D camera location in relation a world surface model.

        This method approximates the relative location of a camera using the intersection of two lines
        projected along the z-plane in image space from two separate locations.

        The camera intrinsics (self.camera_matrix, self.distortion_matrix), and extrinsics
        (rotation, translation), must be known, and these are used to project points from the
        world model into the image space.

        The z-plane projections should intersect at a location image space, and we can use this
        location in world space to estimate the 2d camera location.

        Note, this is an approximate 2d estimate, and should not be considered equivelant to
        a 3d pose that we might estimate using a conventional checkerboard.

        In other words, it's a quick and nasty 2d estimate that is sufficient for approximating
        the camera position for use in cropping rotated image crops from within the camera view.

        Args:
            world_model (WorldModel): pose-initialised camera with valid extrinsics.

        Returns:
            (x,y): Returns world coordinates.
        """
        assert len(world_model.model_points >= 2), "Not enough model points to estimate camera point"
        assert len(world_model.image_points >= 2), "Not enough model points to estimate camera point"

        world_points = world_model.model_points

        # TODO - Verify that the world points used to project rays to the camera are the min and max x-values.

        # We select the first 3D world point and project along the z-plane
        pt1 = np.array([[[world_points[0][0], world_points[0][1], 0]]], dtype='float32')
        pt2 = np.array([[[world_points[0][0], world_points[0][1], -world_model.model_scale]]], dtype='float32')

        # Project the 3D world points to 2D image points.
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

        # Repeat the previous step for the second (last) 3D world point.
        # TODO - obviously wrap this into a function or loop
        pt1 = np.array([[[world_points[-1][0], world_points[-1][1], 0]]], dtype='float32')
        pt2 = np.array([[[world_points[-1][0], world_points[-1][1], -world_model.model_scale]]], dtype='float32')

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

        self.camera_2d_image_space_location = ray_intersection(line1, line2)

        return self.camera_2d_image_space_location

    def file_type(self):
        """Probe file type instance searching by MIME type or file extension.

        Returns:
            (str): file extension.
            (str): MIME type.
        """
        if hasattr(self, "filepath"):
            kind = filetype.guess(self.filepath)
            return kind.extension, kind.MIME
        else:
            return None, None

    def save_frame(self, dest_path='./image.png'):
        """Grabs a frame and saves it to file.

        Args:
            dest_path (str): destination for saved image.
        """
        frame = 0

        if hasattr(self, "set_position"):
            self.set_position(frame_number=100)

        _frame = self.get_frame()

        cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB, _frame)
        img = Image.fromarray(_frame)
        img.save(dest_path)

        if self.verbose_mode:
            print("Saved frame {0} to {1}".format(frame, dest_path))

    def close(self):
        if self.video_object is not None:
            print(self.__class__, "Closing video object...")
            self.video_object.release()

    def __repr__(self):
        """Overriding repr

        Returns:
            (str): A string representation of the object
        """
        return '{} | {}'.format(self.__class__, self.__hash__())

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return self.__str__()
