"""Camera controller for multiple camera view stitching"""
from cameras import VKCamera
from cameras.helpers.panorama import *
# from cameras.helpers.camera_parser import load_annotations_from_json


class VKCameraPanorama(VKCamera):

    def __init__(self, input_camera_models, stitch_params=None,
                 panorama_projection_models=None,
                 surface_name=None,
                 annotations=None,
                 verbose_mode=False):
        """Constructor for panoramic image class.  Rather than dealing explicitly with
        images, this class handles camera models that should be instantiated by the
        owner of this class.

        NB: No camera-wise calibration is required.  Generally, we would calibrate
        the panoramic composite with respect to world coordinates.

        Args:
            input_camera_models (list): A list of core camera objects.
            stitch_params (dict): Overrides default stitching params.
            panorama_projection_models (list): List of camera projection models (These are
            normally derived from the <compute_transforms> function, but since they are
            non-detirministic, the user may have refined previously optimal set of
            projections which should override any new set obtained at init.
            verbose_mode (bool): Additional class detail logging.
        """

        super().__init__(surface_name=surface_name, verbose_mode=verbose_mode)

        self.input_camera_models = input_camera_models
        self.input_names = []

        _input_images = []

        for idx, camera in enumerate(input_camera_models):
            _img = camera.get_frame()
            _input_images.append(_img)
            self.input_names.append("Camera {0}".format(idx))

        # Default parameters.
        params = stitch_params or {"work_megapix": 0.6,
                                   "warp_type": VK_PANORAMA_WARP_SPHERICAL,
                                   "wave_correct": "horiz",
                                   "blend_type": VK_PANORAMA_BLEND_MULTIBAND,
                                   "feature_match_algorithm": VK_PANORAMA_FEATURE_BRISK,
                                   "blend_strength": 5}

        # Initiate the stitching class.
        self._stitching = VKPanorama(params=params)

        # TODO - solve metric import issue where the conversion from list to ndarray seems to be discretising the element values.
        # if panorama_projection_models is None:
        if True:
            # Compile the input-wise panoramic projection matrices.
            # This can take a few seconds for large composites, so we do it here once only and retain the matrices
            # for future use.
            print("Building panorama for the first time...")
            _composite_image, self.panorama_projection_models = self._stitching.compute_transforms(input_images=_input_images, input_names=self.input_names)
        # else:
        #     # Alternately, if a serialised set of projection models are available, we use them.
        #     print("Building panorama from json...")
        #     for m in panorama_projection_models:
        #         print(m)
        #     _composite_image = self._stitching.stitch(camera_models=panorama_projection_models, input_images=_input_images)

        # Composite image properties.
        self._width = _composite_image.shape[1]
        self._height = _composite_image.shape[0]

        # Retain parameters
        self.stitching_parameters = params

        # Retain annotations for demonstration purposes.
        self.annotations = annotations

    def frame_position(self):
        """The current frame number in the video resource.

        Returns:
            (list): The CAP_PROP_POS_FRAMES property in a list over each camera instance.
        """
        _positions = []
        for camera in self.input_camera_models:
            _positions.append(camera.frame_position)
        return _positions

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
            _frame_number = camera.frame_position()

        that = None
        if self.annotations is not None:
            that = list((item for item in self.annotations if item['Frame'] == _frame_number))

        frame = self._stitching.stitch(panorama_projection_models=self.panorama_projection_models, input_images=_input_images, camera_models=self.input_camera_models, annotations=that)

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

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return self._stitching
