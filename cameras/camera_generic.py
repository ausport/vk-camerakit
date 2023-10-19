"""Camera controller for video capture from generic video camera (webcam)"""
import cv2
import math
from cameras import VKCamera


class VKCameraGenericDevice(VKCamera):

    def __init__(self, device=0, configs=None, verbose_mode=False, surface_name=None):
        super().__init__(configs=configs, surface_name=surface_name, verbose_mode=verbose_mode)

        print(f"Searching for generic OpenCV-compatible capture device at {device}")

        self.video_object = cv2.VideoCapture(device)
        if self.video_object.isOpened():
            self.device = device

        if configs is not None:
            self.set_capture_parameters(configs)

    def eof(self):
        """Overrides eof.

        Returns:
            (bool): True if video device is available.
        """
        return not self.video_object.isOpened()

    def get_frame(self):
        res, frame = self.video_object.read()
        # Pillow assumes RGB - OpenCV reads BRG
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        return frame

    def exposure_time(self):
        """Current exposure time of the camera. (if available)

        Returns:
            (int): Exposure time.
        """
        return self.video_object.get(cv2.CAP_PROP_EXPOSURE) # / 1e3

    def set_capture_parameters(self, configs):
        """Updates capture device properties.
        The default instance of this method assumes the capture device is OpenCV-compatible.
        This method should be overridden for other devices (e.g. Vimba-compatible IP cameras).

        Args:
            configs (dict): dictionary of configurations.  The keys are expected to be consistent with OpenCV flags.

        Returns:
            (int): Success.
        """
        assert type(configs) is dict, "WTF!!  set_capture_parameters: A dict was expected but not received..."
        result = True

        if "CAP_PROP_FRAME_WIDTH" in configs:
            result = result and self.video_object.set(cv2.CAP_PROP_FRAME_WIDTH, int(configs["CAP_PROP_FRAME_WIDTH"]))
        if "CAP_PROP_FRAME_HEIGHT" in configs:
            result = result and self.video_object.set(cv2.CAP_PROP_FRAME_HEIGHT, int(configs["CAP_PROP_FRAME_HEIGHT"]))
        if "CAP_PROP_FPS" in configs:
            result = result and self.video_object.set(cv2.CAP_PROP_FPS, int(configs["CAP_PROP_FPS"]))
        # if "CAP_PROP_EXPOSURE" in configs:
        #     result = result and self.video_object.set(cv2.CAP_PROP_EXPOSURE, int(configs["CAP_PROP_EXPOSURE"]))

        # We need to manually set the FPS to it's maximum in the case that we've previously changed to a
        # higher resolution (which automatically drops the fps).
        self.video_object.set(cv2.CAP_PROP_FPS, math.inf)

        return result

    def name(self):
        return '{} | {}'.format("Generic", self.device)

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return "\nGeneric OpenCV-Compatible Camera Source:" \
               "\n\tVideo Device     : {0}" \
               "\n\tWidth            : {1}" \
               "\n\tHeight           : {2}" \
               "\n\tFrame Rate       : {3}" \
               "\n\tFrame Count      : {4}".format(self.device,
                                                   self.width(),
                                                   self.height(),
                                                   self.fps(),
                                                   self.frame_count())
