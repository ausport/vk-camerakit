"""Camera controller for BlackMagic BRAW format image/video file resource"""
from cameras import VKCamera
from pybraw import PixelFormat, ResolutionScale
from pybraw.torch.reader import FrameImageReader
from torchvision import transforms

import numpy as np


class VKCameraBlackMagicRAW(VKCamera):
    def __init__(self, filepath, verbose_mode=False, surface_name=None, device="cuda", scale=ResolutionScale.Eighth):
        super().__init__(surface_name=surface_name, verbose_mode=verbose_mode)

        self.video_object = FrameImageReader(filepath, processing_device=device)
        self._filepath = filepath
        self._processing_device = device
        self._resolution_scale = scale
        self._current_frame_index = 0

    def get_frame(self, frame_number=None):
        """Raw camera file image.

        Args:
            frame_number (int): (optional) seek to frame number, then grab frame

        Returns:
            (array): image.
        """
        self._current_frame_index = frame_number or self._current_frame_index
        transform = transforms.ToPILImage()

        with self.video_object.run_flow(PixelFormat.RGB_F32_Planar, max_running_tasks=3) as task_manager:
            task = task_manager.enqueue_task(self._current_frame_index, resolution_scale=self._resolution_scale)
            image_tensor = task.consume()
            # Increment frame index
            if self._current_frame_index < self.frame_count():
                self._current_frame_index += 1

            return np.asarray(transform(image_tensor))

    def set_position(self, frame_number):
        """Seek to frame number

        Args:
            frame_number (int): valid frame number for assignment.
        Returns:
            None
        """
        if frame_number < self.frame_count():
            self._current_frame_index = frame_number

    def frame_position(self):
        """The current frame number in the video resource.

        Returns:
            (int): The CAP_PROP_POS_FRAMES property - zero if a live camera.
        """
        return self._current_frame_index

    def frame_count(self):
        """The number of frames in the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_COUNT property - zero if a live camera.
        """
        return self.video_object.frame_count()

    def width(self):
        """The pixel width of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_WIDTH property.
        """
        return self.video_object.frame_width()

    def height(self):
        """The pixel height of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_HEIGHT property.
        """
        return self.video_object.frame_height()

    def fps(self):
        """The frames per second of the video resource.

        Returns:
            (float): The CAP_PROP_FPS property.
        """
        return self.video_object.frame_rate()

    def camera_type(self):
        """The frames per second of the video resource.

        Returns:
            (float): The CAP_PROP_FPS property.
        """
        return self.video_object.camera_type()

    def eof(self):
        """Signals end of video file.

        Returns:
            (bool): True if end of file.
        """
        return False

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return "Blackmagic BRAW File Source:" \
               "\n\tFilename         : {0}" \
               "\n\tCamera Type      : {1}" \
               "\n\tWidth            : {2}" \
               "\n\tHeight           : {3}" \
               "\n\tFrame Rate       : {4}" \
               "\n\tFrame Count      : {5}".format(self._filepath,
                                                   self.camera_type(),
                                                   self.width(),
                                                   self.height(),
                                                   self.fps(),
                                                   self.frame_count())
