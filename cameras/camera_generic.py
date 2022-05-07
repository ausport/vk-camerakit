"""Camera controller for video capture from generic video camera (webcam)"""
import cv2
import math
from cameras import VKCamera


class VKCameraGenericDevice(VKCamera):

    def __init__(self, device=0, verbose_mode=False, surface_name=None):
        super().__init__(surface_name=surface_name, verbose_mode=verbose_mode)

        self.video_object = cv2.VideoCapture(device)
        if self.video_object.isOpened():
            self.device = device

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

        # We need to manually set the FPS to it's maximum in the case that we've previously changed to a
        # higher resolution (which automatically drops the fps).
        self.video_object.set(cv2.CAP_PROP_FPS, math.inf)

        return result

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return "\nCamera Source:" \
               "\n\tVideo Device     : {0}" \
               "\n\tWidth            : {1}" \
               "\n\tHeight           : {2}" \
               "\n\tFrame Rate       : {3}" \
               "\n\tFrame Count      : {4}".format(self.device,
                                                   self.width(),
                                                   self.height(),
                                                   self.fps(),
                                                   self.frame_count())
