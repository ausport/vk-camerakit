"""Camera controller for BlackMagic BRAW format image/video file resource"""
from cameras import VKCamera
from pybraw import PixelFormat, ResolutionScale
from pybraw.torch.reader import FrameImageReader
from pybraw import _pybraw, verify

from PIL import Image
import numpy as np


class MyCallback(_pybraw.BlackmagicRawCallback):
    def ReadComplete(self, job, result, frame):
        frame.SetResourceFormat(_pybraw.blackmagicRawResourceFormatRGBAU8)
        process_job = verify(frame.CreateJobDecodeAndProcessFrame())
        verify(process_job.Submit())
        process_job.Release()

    def ProcessComplete(self, job, result, processed_image):
        self.processed_image = processed_image


class VKCameraBlackMagicRAW(VKCamera):
    def __init__(self, filepath, verbose_mode=False, surface_name=None, with_cuda=True):
        super().__init__(surface_name=surface_name, verbose_mode=verbose_mode)

        self.filepath = filepath
        self.with_cuda = with_cuda

        # Asynchronous GPU decoding
        # self.video_object = FrameImageReader(filepath, processing_device='cuda')

        factory = _pybraw.CreateBlackmagicRawFactoryInstance()
        self._codec = verify(factory.CreateCodec())
        self.video_object = verify(self._codec.OpenClip(filepath))

        self._callback = MyCallback()
        verify(self._codec.SetCallback(self._callback))

    def get_frame(self, frame_number=None):
        """Raw camera file image.

        Args:
            frame_number (int): (optional) seek to frame number, then grab frame

        Returns:
            (array): image.
        """
        f = frame_number or self.frame_position()
        print("getting {0}".format(f))
        read_job = verify(self.video_object.CreateJobReadFrame(f))
        read_job.Submit()
        read_job.Release()

        verify(self._codec.FlushJobs())

        resource_type = verify(self._callback.processed_image.GetResourceType())
        assert resource_type == _pybraw.blackmagicRawResourceTypeBufferCPU
        np_image = self._callback.processed_image.to_py()
        del self._callback.processed_image

        pil_image = Image.fromarray(np_image[..., :3])
        print(pil_image)
        return np.asarray(pil_image)

    def set_position(self, frame_number):
        """Seek to frame number

        Args:
            frame_number (int): valid frame number for assignment.
        Returns:
            None
        """
        # self.video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        pass

    def frame_position(self):
        """The current frame number in the video resource.

        Returns:
            (int): The CAP_PROP_POS_FRAMES property - zero if a live camera.
        """
        # print(verify(self.video_object.GetTimecodeForFrame()))
        # return verify(self.video_object.GetTimecodeForFrame())
        return 1

    def frame_count(self):
        """The number of frames in the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_COUNT property - zero if a live camera.
        """
        return verify(self.video_object.GetFrameCount())

    def width(self):
        """The pixel width of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_WIDTH property.
        """
        return verify(self.video_object.GetWidth())

    def height(self):
        """The pixel height of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_HEIGHT property.
        """
        return verify(self.video_object.GetHeight())

    def fps(self):
        """The frames per second of the video resource.

        Returns:
            (float): The CAP_PROP_FPS property.
        """
        return verify(self.video_object.GetFrameRate())

    def camera_type(self):
        """The frames per second of the video resource.

        Returns:
            (float): The CAP_PROP_FPS property.
        """
        return verify(self.video_object.GetCameraType())

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
               "\n\tFilename         : {1}" \
               "\n\tWidth            : {2}" \
               "\n\tHeight           : {3}" \
               "\n\tFrame Rate       : {4}" \
               "\n\tFrame Count      : {5}".format(self.filepath,
                                                   self.camera_type(),
                                                   self.width(),
                                                   self.height(),
                                                   self.fps(),
                                                   self.frame_count())
