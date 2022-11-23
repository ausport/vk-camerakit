"""Camera controller for multiple camera view stitching"""
from cameras import VKCamera
from cameras.helpers.panorama import *


class VKCameraPanorama(VKCamera):

    def __init__(self, input_camera_models,
                 stitch_params=None,
                 surface_name=None,
                 verbose_mode=False):
        """Constructor for panoramic image class.  Rather than dealing explicitly with
        images, this class handles camera models that should be instantiated by the
        owner of this class.

        NB: No camera-wise calibration is required.  Generally, we would calibrate
        the panoramic composite with respect to world coordinates.

        Args:
            input_camera_models (list): A list of VKCamera camera objects.
            stitch_params (dict): Overrides default stitching params.
            surface_name (str): name of the calibrated surface model.
            verbose_mode (bool): Additional class detail logging.
        """

        super().__init__(surface_name=surface_name, verbose_mode=verbose_mode)

        self.input_camera_models = input_camera_models
        self.input_names = []

        _input_images = []

        for idx, camera in enumerate(input_camera_models):
            # NB - for consistency between panorama results, we always use the first frame to construct matching features.
            camera.set_position(frame_number=1)
            _img = camera.get_frame()
            _input_images.append(_img)
            self.input_names.append("Camera {0}".format(idx))

        # Default parameters if none are passed.
        stitch_params = stitch_params or {"work_megapix": 0.3,
                                          "warp_type": VK_PANORAMA_WARP_SPHERICAL,
                                          "wave_correct": "horiz",
                                          "blend_type": VK_PANORAMA_BLEND_MULTIBAND,
                                          "feature_match_algorithm": VK_PANORAMA_FEATURE_BRISK,
                                          "blend_strength": 0.25}

        # Initiate the stitching controller class.
        self._stitching_controller = VKPanoramaController(params=stitch_params)

        # Retain parameters
        self.stitching_parameters = stitch_params

        # Compile the input-wise panoramic projection matrices.
        # This can take a few seconds for large composites, so we do it here once only and retain the matrices
        # for future use.
        print("Building panorama for the first time...")

        while True:
            _composite_image, self.panorama_projection_models = self._stitching_controller.compute_transforms(input_images=_input_images, input_names=self.input_names)
            if _composite_image is not None:
                break

        # Composite image properties.
        self._width = _composite_image.shape[1]
        self._height = _composite_image.shape[0]

    def frame_position(self):
        """The current frame number in the video resource.

        Returns:
            (list): The CAP_PROP_POS_FRAMES property in a list over each camera instance.
        """
        _positions = []
        for camera in self.input_camera_models:
            _positions.append(camera.frame_position())
        return _positions[0]

    def frame_count(self):
        """The number of frames in the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_COUNT property - zero if a live camera.
        """
        _frames = math.inf
        for camera in self.input_camera_models:
            _frames = min(_frames, camera.frame_count())

        return _frames

    def set_position(self, frame_number):
        """Seek to frame number over all input cameras

        Args:
            frame_number (int): valid frame number for assignment.
        Returns:
            None
        """
        for idx, camera in enumerate(self.input_camera_models):
            camera.set_position(frame_number=frame_number)

    def fps(self):
        """The frames per second of the video resource.

        Returns:
            (float): The CAP_PROP_FPS property.
        """
        _fps = []
        for idx, camera in enumerate(self.input_camera_models):
            _fps.append(camera.fps())
        return float(np.mean(_fps))

    def get_frame(self):
        """Panoramic camera image.

        Returns:
            (array): panoramic-scale image.
        """
        _input_images = []
        _frame_number = 0

        # Now we have a working stitcher, it should be faster.
        for idx, camera in enumerate(self.input_camera_models):
            _img = camera.get_frame()
            _input_images.append(_img)

        frame = self._stitching_controller.stitch(panorama_projection_models=self.panorama_projection_models, input_images=_input_images)

        return frame

    def eof(self):
        """Overrides eof.

        Returns:
            (bool): True if video device is available.
        """
        _result = False
        for idx, camera in enumerate(self.input_camera_models):
            if not camera.video_object.isOpened():
                _result = True
        return _result

    def width(self):
        """The pixel width of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_WIDTH property.
        """
        return int(self._width)

    def height(self):
        """The pixel height of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_HEIGHT property.
        """
        return int(self._height)

    def projected_panoramic_point_for_2d_world_point(self, world_point):
        """Estimate 2d panoramic coordinates from 2d world coordinates.
        Args:
            world_point (x, y): world/model coordinates.

        Returns:
            (x,y): Returns world coordinates.
        """
        panoramic_image_point = self._stitching_controller.panoramic_point_for_world_point(world_point=world_point,
                                                                                           panorama_projection_models=self.panorama_projection_models,
                                                                                           camera_models=self.input_camera_models)
        return panoramic_image_point

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

        # Panoramic parameters
        if self.__class__.__name__ == "VKCameraPanorama":
            assert hasattr(self, "input_camera_models"), "Panoramic model doesn't include input camera models..."
            assert hasattr(self, "stitching_parameters"), "Panoramic model doesn't include stitching parameters..."
            assert hasattr(self, "panorama_projection_models"), "Panoramic model doesn't include panorama projection parameters..."

            _camera_parameters.update({'stitching_parameters': self.stitching_parameters})
            _pano_camerawise_models = []

            for idx, input_camera in enumerate(self.input_camera_models):
                assert input_camera.__class__.__name__ != "VKCameraPanorama", "This would be bad..."
                assert len(self.input_camera_models) == len(self.panorama_projection_models), "This would also be bad..."

                projection_model = self.panorama_projection_models[idx]

                projection_model_parameters = {
                    "name": projection_model["name"],
                    "short_name": projection_model["short_name"],
                    "corner": projection_model["corner"],
                    "rotation": projection_model["rotation"].tolist(),
                    "extrinsics": projection_model["extrinsics"].tolist()
                }

                # Compile json representation of camera model and projection parameters
                _pano_camerawise_models.append(
                    {"input_camera": projection_model["short_name"],
                     "input_camera_model": input_camera.camera_model_json(),
                     "projection_model_parameters": projection_model_parameters})

            _camera_parameters.update({'panorama_projection_models': _pano_camerawise_models})

        # World model parameters
        if hasattr(self.surface_model, "homography"):
            _camera_parameters.update({'homography': self.surface_model.homography.tolist()})
        if hasattr(self.surface_model, "image_points"):
            _camera_parameters.update({'image_points': self.surface_model.image_points.tolist()})
        if hasattr(self.surface_model, "model_points"):
            _camera_parameters.update({'model_points': self.surface_model.model_points.tolist()})

        return _camera_parameters

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return self.__class__.__name__
