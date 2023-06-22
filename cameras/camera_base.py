"""Generic class for all image sources"""
import time

import cv2
import filetype
import json
import numpy as np
import os
from PIL import Image

from world_models.geometry import ray_intersection
from world_models import world_model as surface
import cameras

VK_CAPTURE_MODE_PREVIEW = 0
VK_CAPTURE_MODE_RECORD = 1


class VKCamera:
    def __init__(self, surface_name=None, verbose_mode=False):
        """Constructor for generic image source class.
        Use subclassed instances of this class

        Args:
            surface_name (str): Surface model name.
            verbose_mode (bool): Additional class detail logging.

        Usage:
            Subclass VKCamera with optional world surface:
                e.g. camera = VKCameraVideoFile(filepath, surface_name)
                e.g. camera = VKCameraGenericDevice(device, surface_name)

            Alternately, pre-configured json format camera files may include
            camera intrinsics and extrinsics, world homographies,
            camera capture properties, world-image correspondences, and a class type.

            In the case of file-based objects (e.g. VKCameraVideoFile), the filename
            can be either a video/image resource, or a json-formatted configuration.
            If the former, default camera properties are assumed.

        """

        self.verbose_mode = verbose_mode
        self.video_object = None

        # Camera intrinsics properties
        self.focal_length = 23.
        self.camera_matrix = np.zeros((3, 3))
        self.distortion_matrix = np.zeros((4, 1))

        # Camera extrinsic properties
        self.rotation_vector = np.zeros((3, 3))
        self.translation_vector = np.zeros((3, 3))
        self.camera_2d_image_space_location = None

        self.image_rotation = cameras.VK_ROTATE_NONE

        # Surface model
        self.surface_model = None
        if surface_name is not None:
            self.surface_model = surface.VKWorldModel(sport=surface_name)

        self.capture_mode = VK_CAPTURE_MODE_PREVIEW
        self._video_writer = None

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
            (float): The CAP_PROP_FPS property.
        """
        return float(self.video_object.get(cv2.CAP_PROP_FPS))

    def image_rotation(self):
        """The image rotation vector.

        Returns:
            (int): The CAP_PROP_ROTATION property.
        """
        return self.image_rotation

    def set_image_rotation(self, rotate):
        """
        The function cv::rotate rotates the array in one of three different ways:
        *   Rotate by 90 degrees clockwise (rotateCode = ROTATE_90_CLOCKWISE = 0).
        *   Rotate by 180 degrees clockwise (rotateCode = ROTATE_180 = 1).
        *   Rotate by 270 degrees clockwise (rotateCode = ROTATE_90_COUNTERCLOCKWISE = 2).
        self._most_recent_image = cv2.rotate(self._most_recent_image, self._rotation_vector)
        """
        assert -1 <= rotate <= 2, "Invalid rotation vector..."
        self.image_rotation = rotate

    def eof(self):
        """Signals end of video file.

        Returns:
            (bool): True if end of file.
        """
        return int(self.video_object.get(cv2.CAP_PROP_POS_FRAMES)) >= int(self.video_object.get(cv2.CAP_PROP_FRAME_COUNT))

    def is_available(self):
        """Returns the current status of an imaging device.
        NB: Non-imaging camera classes (file-based) will raise an exception.

        Returns:
            (bool): True if imaging device is available.
        """
        if self.__class__.__name__ in ["VKCameraGenericDevice", "VKCameraVimbaDevice"]:
            return self.video_object.isOpened()
        else:
            raise NotImplementedError

    def surface_model(self):
        return self.surface_model

    def undistorted_image(self, image=None):
        """Undistorted input image with camera matrices.

        Args:
            image (array): (optional) input image.

        Returns:
            (array): Distortion-corrected image.
        """
        image = image or self.get_frame()
        return cv2.undistort(image, self.camera_matrix, self.distortion_matrix, None, None)

    def update_camera_properties(self, with_distortion_matrix=None, with_camera_matrix=None):
        """Recalculates the camera intrinsics matrix.

        Args:
            with_distortion_matrix (array): (optional) 4x1 numpy array defining camera distortion.
            with_camera_matrix (array): (optional) 3x3 numpy array defining camera dimension matrix.
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


    def instantiate_writer_with_path(self, path):
        """Initialise an OpenCV file writer at the specified path.

        Args:
            path (str): destination for saved video.
        Returns:
            (cv2.VideoWriter): an initialised video writer object.
        """
        assert os.path.exists(os.path.dirname(path)), "Can't instantiate a video writer to a non-existent path."

        FOURCC = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        return cv2.VideoWriter(path, FOURCC, self.fps(), (self.width(), self.height()), True)


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
        _frame = self.get_frame()
        img = Image.fromarray(_frame)
        img.save(dest_path)

    def set_capture_mode(self, mode):
        """External method to stop record mode.

        Args:
            mode (int): VK_CAPTURE_MODE_RECORD or VK_CAPTURE_MODE_PREVIEW
        Returns:
            None
        """
        assert mode == VK_CAPTURE_MODE_RECORD or mode == VK_CAPTURE_MODE_PREVIEW
        self.capture_mode = mode

    def save_video(self, video_export_path, size=(1920,1080), fps=25):
        """Saves current camera model imagery in mp4 format.

        Args:
            video_export_path (str): destination path.
            size(tuple): output size - default 1920 x 1080
            fps(float): desired frame rate.
        Returns:
            None
        """
        if os.path.exists(os.path.split(video_export_path)[0]):
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            # pad out to a 16:9 aspect ratio
            new_image_width = self.width()
            new_image_height = int(new_image_width / 16) * 9
            self._video_writer = cv2.VideoWriter(str(video_export_path), fourcc, fps, size, True)

            self.capture_mode = VK_CAPTURE_MODE_RECORD
            from threading import Thread

            thread = Thread(target=self.cap_loop)
            # thread.daemon = True
            thread.start()
            time.sleep(2)
            self.capture_mode = VK_CAPTURE_MODE_PREVIEW
            # thread.join()

            # while self.capture_mode == VK_CAPTURE_MODE_RECORD:
            #     _frame = self.get_frame()
            #     cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB, _frame)
            #
            #     # old_image_height, old_image_width, channels = _frame.shape
            #     # _padded = np.full((new_image_height, new_image_width, channels), (0, 0, 0), dtype=np.uint8)
            #     #
            #     # # compute center offset
            #     # x_center = (new_image_width - old_image_width) // 2
            #     # y_center = (new_image_height - old_image_height) // 2
            #     #
            #     # # copy img image into center of result image
            #     # _padded[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = _frame
            #     # _padded = cv2.cvtColor(np.array(_padded), cv2.COLOR_RGB2BGR)
            #     # _padded = cv2.resize(_padded, dsize=size, interpolation=cv2.INTER_CUBIC)
            #     _video_writer.write(_frame)
            #
            #     if self.eof():
            #         break

    def cap_loop(self):
        print("Starting queue")
        while self.capture_mode == VK_CAPTURE_MODE_RECORD:
            _frame = self.get_frame()
            self._video_writer.write(_frame)
            print(time.time())
        print("Exiting queue")
        self._video_writer.release()

    def export_json(self, json_path):
        """Export current camera model in json format.

        Args:
            json_path (str): destination path.

        Returns:
            None
        """

        j = json.dumps(
            self.camera_model_json(),
            indent=4,
            separators=(',', ': ')
        )

        if not json_path.endswith(".json"):
            json_path = json_path + ".json"

        with open(json_path, 'w') as data_file:
            data_file.write(j)

    def camera_model_json(self):
        """Serialise the existing model parameters.
        Note that we store all of the world model parameters here too.
        Deserialisation should return a configured surface model.

        Returns:
            Serialised camera model.
        """

        _camera_parameters = {'class': self.__class__.__name__}

        # Camera-specific parameters
        if hasattr(self, "filepath"):
            _camera_parameters.update({'image_path': self.filepath})
        if hasattr(self, "surface_model"):
            if self.surface_model is not None:
                _camera_parameters.update({'surface_model': self.surface_model.surface_model_name()})
        if hasattr(self, "focal_length"):
            _camera_parameters.update({'focal_length': self.focal_length})
        if hasattr(self, "camera_matrix"):
            _camera_parameters.update({'camera_matrix': self.camera_matrix.tolist()})
        if hasattr(self, "distortion_matrix"):
            _camera_parameters.update({'distortion_matrix': self.distortion_matrix.tolist()})
        if hasattr(self, "rotation_vector"):
            _camera_parameters.update({'rotation_vector': self.rotation_vector.tolist()})
        if hasattr(self, "translation_vector"):
            _camera_parameters.update({'translation_vector': self.translation_vector.tolist()})
        # Image transform parameters
        if hasattr(self, "image_rotation"):
            _camera_parameters.update({'image_rotation': self.image_rotation})

        # World model parameters
        if hasattr(self.surface_model, "homography"):
            _camera_parameters.update({'homography': self.surface_model.homography.tolist()})
        if hasattr(self.surface_model, "image_points"):
            _camera_parameters.update({'image_points': self.surface_model.image_points.tolist()})
        if hasattr(self.surface_model, "model_points"):
            _camera_parameters.update({'model_points': self.surface_model.model_points.tolist()})

        return _camera_parameters

    def close(self):
        if self.video_object is not None:
            print(self.__class__, "Closing video object...")
            self.video_object.release()

    def name(self):
        return self.__repr__()

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
