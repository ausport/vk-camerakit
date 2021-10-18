"""Camera controller for multiple camera view stitching"""
from cameras import VKCamera
from cameras.helpers.panorama import *


class VKCameraPanorama(VKCamera):

    def __init__(self, input_cameras, stitch_params=None, panorama_projection_models=None,  surface_name=None, verbose_mode=False):
        """Constructor for panoramic image class.  Rather than dealing explicitly with
        images, this class handles camera models that should be instantiated by the
        owner of this class.

        NB: No camera-wise calibration is required.  Generally, we would calibrate
        the panoramic composite with respect to world coordinates.

        Args:
            input_cameras (list): Requires a list of string paths.
            stitch_params (dict): Overrides default stitching params.
            panorama_projection_models (list): List of camera projection models (These are
            normally derived from the <compute_transforms> function, but since they are
            non-detirministic, the user may have refined previously optimal set of
            projections which should override any new set obtained at init.
            verbose_mode (bool): Additional class detail logging.
        """

        super().__init__(surface_name=surface_name, verbose_mode=verbose_mode)

        self.input_cameras = input_cameras
        self.input_names = []

        _input_images = []

        for idx, camera in enumerate(input_cameras):
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

        # if panorama_projection_models is None:
        # TODO - solve metric import issue where the conversion from list to ndarray seems to be discretising the element values.
        if True:
            # Compile the input-wise panoramic projection matrices.
            # This can take a few seconds for large composites, so we do it here once only and retain the matrices
            # for future use.
            print("Building panorama for the first time...")
            _composite_image, self.panorama_projection_models = self._stitching.compute_transforms(input_images=_input_images, input_names=self.input_names)
        else:
            # Alternately, if a serialised set of projection models are available, we use them.
            print("Building panorama from json...")
            for m in panorama_projection_models:
                print(m)
            _composite_image = self._stitching.stitch(camera_models=panorama_projection_models, input_images=_input_images)

        # for model in self.panorama_projection_models:
        #     print(model)

        # Composite image properties.
        self._width = _composite_image.shape[1]
        self._height = _composite_image.shape[0]

        # Retain parameters
        self.stitching_parameters = params

    def frame_count(self):
        """The number of frames in the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_COUNT property - zero if a live camera.
        """
        _frames = math.inf
        for camera in self.input_cameras:
            _frames = min(_frames, camera.frame_count())

        return _frames

    def set_position(self, frame_number):
        """Seek to frame number over all input cameras

        Args:
            frame_number (int): valid frame number for assignment.
        Returns:
            None
        """
        for idx, camera in enumerate(self.input_cameras):
            camera.set_position(frame_number=frame_number)

    def get_frame(self):
        """Panoramic camera image.

        Returns:
            (array): panoramic-scale image.
        """
        _input_images = []
        # Now we have a working stitcher, it should be faster.
        for idx, camera in enumerate(self.input_cameras):
            _img = camera.get_frame()
            _input_images.append(_img)

        # TODO - any annotations should be added at this stage..  Add annotations dict as parameter to be parsed.
        # that = list((item for item in annotations if item[0] == '{0}'.format(frame_number)))
        frame = self._stitching.stitch(camera_models=self.panorama_projection_models, input_images=_input_images, annotations=None)

        return frame

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
