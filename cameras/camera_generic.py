"""Camera controller for video capture from generic video camera (webcam)"""
import cv2
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
