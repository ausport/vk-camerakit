"""Camera controller for existing image/video file resource"""
import cv2
from cameras import CameraSource


class CameraFile(CameraSource):
    def __init__(self, filepath, verbose_mode=False):
        super().__init__(verbose_mode=verbose_mode)

        self.video_object = cv2.VideoCapture(filepath)
        if self.video_object.isOpened():
            self.filepath = filepath

    def get_frame(self, frame_number=None):
        if frame_number is not None:
            self.video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

        res, frame = self.video_object.read()
        # Pillow assumes RGB - OpenCV reads BRG
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        return frame

    def set_position(self, frame_number=100):
        self.video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)


