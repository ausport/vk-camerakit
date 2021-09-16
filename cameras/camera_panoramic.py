"""Camera controller for multiple camera view stitching"""
import cv2
from cameras import VKCamera
from cameras.helpers.panorama import *


class VKCameraPanorama(VKCamera):

    def __init__(self, input_files, stitch_params=None, verbose_mode=False):
        """Constructor for panoramic image class.

        Args:
            input_files (list): Requires a list of string paths.
            verbose_mode (bool): Additional class detail logging.
        """

        super().__init__(verbose_mode=verbose_mode)

        self._input_files = _file_list
        _input_images = []

        for file in self._input_files:
            _capture = cv2.VideoCapture(file)
            success, img = _capture.read()
            assert success, "Couldn't open {0}".format(file)
            _input_images.append(img)

        # Default parameters.
        params = stitch_params or {"work_megapix": 0.6,
                                   "warp_type": VK_PANORAMA_WARP_SPHERICAL,
                                   "wave_correct": "horiz",
                                   "blend_type": VK_PANORAMA_BLEND_MULTIBAND,
                                   "feature_match_algorithm": VK_PANORAMA_FEATURE_BRISK,
                                   "blend_strength": 5}

        # Initiate the stitching class.
        self._stitching = VKPanorama(params=params)

        # Compile the input-wise panoramic projection matrices.
        # This can take a few seconds for large composites, so we do it here once only and retain the matrices
        # for future use.
        _composite_image, self._panorama_projection_models = self._stitching.compute_transforms(input_images=_input_images, input_names=self._input_files)

        # Composite image properties.
        self._width = _composite_image.shape[1]
        self._height = _composite_image.shape[0]

    def get_frame(self, frame_number=None):

        # if frame_number is not None:
        #     self.video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

        _input_images = []
        # Now we have a working stitcher, it should be faster.
        for file in self._input_files:
            _capture = cv2.VideoCapture(file)
            _, img = _capture.read()
            assert _, "Couldn't open {0}".format(file)
            _input_images.append(img)

        # TODO - any annotations should be added at this stage..  Add annotations dict as parameter to be parsed.
        # that = list((item for item in annotations if item[0] == '{0}'.format(frame_number)))
        frame = self._stitching.stitch(camera_models=self._panorama_projection_models, input_images=_input_images, annotations=None)

        if frame is not None:
            # Pillow assumes RGB - OpenCV reads BRG
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)

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
