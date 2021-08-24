"""Camera controller for video capture from generic video camera (webcam)"""
import cv2
from cameras import CameraSource


class CameraGeneric(CameraSource):

    def __init__(self, device=0, verbose_mode=False):
        super().__init__(verbose_mode=verbose_mode)

        self.video_object = cv2.VideoCapture(device)
        if self.video_object.isOpened():
            self.filepath = "video device {0}".format(device)
            self.device = device

    def get_frame(self):
        res, frame = self.video_object.read()
        # Pillow assumes RGB - OpenCV reads BRG
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        return frame
