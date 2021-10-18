"""Camera controller for existing image/video file resource"""
import cv2
from cameras import VKCamera


class VKCameraVideoFile(VKCamera):
    def __init__(self, filepath, verbose_mode=False, surface_name=None):
        super().__init__(surface_name=surface_name, verbose_mode=verbose_mode)

        self.video_object = cv2.VideoCapture(filepath)
        if self.video_object.isOpened():
            self.filepath = filepath
            self.is_video = "video" in self.file_type()[1]
        else:
            raise NotImplementedError

    def get_frame(self, frame_number=None):
        """Raw camera file image.

        Args:
            frame_number (int): (optional) seek to frame number, then grab frame

        Returns:
            (array): image.
        """
        if frame_number is not None:
            self.video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

        if self.is_video:
            res, frame = self.video_object.read()
        else:
            # Assume image file type.
            frame = cv2.imread(self.filepath)

        if frame is not None:
            # Pillow assumes RGB - OpenCV reads BRG
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)

        return frame

    def set_position(self, frame_number):
        """Seek to frame number

        Args:
            frame_number (int): valid frame number for assignment.
        Returns:
            None
        """
        self.video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return "Camera Source:" \
               "\n\tFilename         : {0}" \
               "\n\tWidth            : {1}" \
               "\n\tHeight           : {2}" \
               "\n\tFrame Rate       : {3}" \
               "\n\tFrame Count      : {4}".format(self.filepath,
                                                   self.width(),
                                                   self.height(),
                                                   self.fps(),
                                                   self.frame_count())
