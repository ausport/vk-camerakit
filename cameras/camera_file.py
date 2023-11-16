"""Camera controller for existing image/video file resource"""
import queue
import threading
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

        # TODO - load configs if available.
        self.update_camera_properties()

        print(self)

        # Create a frame controller that will loop
        # on a background thread, and accumulate frames to a queue.
        self._frame_queue = queue.Queue()
        self._frame_controller = GenericFrameController(camera=self,
                                                        image_queue=self._frame_queue)
        self._streaming = False

    @property
    def device_id(self):
        return self.filepath

    def get_frame(self, frame_number=None):
        """Raw camera file image.

        Args:
            frame_number (int): (optional) seek to frame number, then grab frame

        Returns:
            (array): image.
        """
        if self.cache_size() > 0:
            return self._frame_controller.next_frame()

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

    def cache_size(self):
        return self._frame_controller.cache_size()

    def pre_roll(self):
        """Pre-rolling is not required for file-based VKCamera subclasses.
        We only keep the stub to retain camera-agnostic code upstream."""
        if not self._streaming:
            self._frame_controller.start()
        self._streaming = True

    def start_streaming(self):
        """An asynchronous image acquisition routine which will queue frames
        streaming from the connected image device."""
        self.pre_roll()

    def stop_streaming(self):
        """Terminate threaded asynchronous image acquisition."""
        self._frame_controller.stop()
        self._streaming = False
    def is_streaming(self):
        """Verifies that the device is streaming."""
        return self._streaming

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
        return "Video File Source:" \
               "\n\tFilename         : {0}" \
               "\n\tWidth            : {1}" \
               "\n\tHeight           : {2}" \
               "\n\tFrame Rate       : {3}" \
               "\n\tFrame Count      : {4}".format(self.filepath,
                                                   self.width(),
                                                   self.height(),
                                                   self.fps(),
                                                   self.frame_count())

class GenericFrameController(threading.Thread):
    def __init__(self, camera: VKCamera, image_queue: queue.Queue):
        threading.Thread.__init__(self)

        self._image_queue = image_queue
        self._lock = threading.Lock()
        self._kill_switch = threading.Event()
        self._camera = camera


    def run(self):
        """Put frames from video to a queue, until the kill switch is thrown."""
        while not self._kill_switch.is_set():
            res, frame = self._camera.video_object.read()
            if res:
                self._image_queue.put_nowait(frame)

    def stop(self):
        self._kill_switch.set()

    def has_queued_frames(self):
        with self._lock:
            return self._image_queue.qsize() > 0

    def cache_size(self):
        with self._lock:
            return self._image_queue.qsize()

    def next_frame(self):
        with self._lock:
            if self._image_queue.empty():
                return None
            else:
                # Pull the image from the queue
                frame = self._image_queue.get_nowait()
                # Unidistort if a distortion matrix is defined.
                return self._camera.undistorted_image(frame)
