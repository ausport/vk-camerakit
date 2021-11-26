"""Camera controller for video capture from generic video camera (webcam)"""
import cv2
from vimba import Vimba, VimbaFeatureError, PixelFormat, VimbaCameraError
from vimba.c_binding import VimbaCError
from cameras import VKCamera


class VKCameraVimbaDevice(VKCamera):

    def __init__(self, ip_address, verbose_mode=False, surface_name=None):
        super().__init__(surface_name=surface_name, verbose_mode=verbose_mode)

        print("Searching for Allied Vision device at {0}".format(ip_address))

        # Get an instance of Vimba
        self.vimba = Vimba.get_instance()

        # Instead of using the context manager (suggested), we manually initialize vimba
        self.vimba.__enter__()

        # Get reference to the camera
        try:
            self.video_object = self.vimba.get_camera_by_id(ip_address)
        except VimbaCameraError:
            raise RuntimeError(f'Failed to access camera: \'{ip_address}\'.')

        # Instead of using the context manager, manually initialize the camera
        self.video_object.__enter__()

        # Keep track of the current frame BEFORE reading from the camera
        self.current_frame = 0

        # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
        try:
            self.video_object.GVSPAdjustPacketSize.run()
            while not self.video_object.GVSPAdjustPacketSize.is_done():
                pass
        except (AttributeError, VimbaFeatureError):
            pass

        # TODO - camera-relevant image capture features
        # Get properties of the camera
        self.camera_name = self.video_object.get_name()
        self.camera_model = self.video_object.get_model()
        self.camera_identifier = self.video_object.get_id()
        self.ip_address = ip_address

        # self.video_object = cv2.VideoCapture(device)
        # if self.video_object.isOpened():
        #     self.device = device

    def eof(self):
        """Overrides eof.

        Returns:
            (bool): True if video device is available.
        """
        return not self.video_object.isOpened()

    def fps(self):
        """The frames per second of the video resource.

        Returns:
            (float): The CAP_PROP_FPS property.
        """
        return float(self.video_object.AcquisitionFrameRateAbs.get())

    def width(self):
        """The pixel width of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_WIDTH property.
        """
        return self.video_object.Width.get()

    def height(self):
        """The pixel height of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_HEIGHT property.
        """
        return self.video_object.Height.get()

    def get_camera_temperature(self):
        """Queries (and returns) the temperature of the camera"""
        return self.video_object.DeviceTemperature.get()

    def get_camera_exposure_time(self):
        """Queries (and returns) the current exposure time of the camera. Returned in milliseconds

        When queried, the result is in microseconds
        """
        return self.video_object.ExposureTimeAbs.get() / 1e3

    def get_frame(self):
        # res, frame = self.video_object.read()
        # # Pillow assumes RGB - OpenCV reads BRG
        # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        # return frame
        return None

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return "\nAllied Vision Camera Source:" \
               "\n\tVideo Device      : {0}" \
               "\n\tIP Address        : {1}" \
               "\n\tDevice Identifier : {2}" \
               "\n\tWidth             : {3}" \
               "\n\tHeight            : {4}" \
               "\n\tTemperature       : {5:.1f} deg C" \
               "\n\tExposure Time     : {6:.1f} ms" \
               "\n\tFrame Rate        : {7:.1f} f.p.s".format(self.camera_model,
                                                   self.ip_address,
                                                   self.camera_identifier,
                                                   self.width(),
                                                   self.height(),
                                                   self.get_camera_temperature(),
                                                   self.get_camera_exposure_time(),
                                                   self.fps())
