"""Camera controller for video capture from Allied Vision video camera (uses Vimba SDK)"""
import cv2
from vimba import Vimba, VimbaFeatureError, PixelFormat, VimbaCameraError
from cameras import VKCamera


class VKCameraVimbaDevice(VKCamera):

    def __init__(self, ip_address="10.2.0.2", verbose_mode=False, surface_name=None):
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

        # Enable auto exposure time setting if camera supports it
        try:
            self.video_object.ExposureAuto.set('Continuous')
        except (AttributeError, VimbaFeatureError):
            pass

        # Enable white balancing if camera supports it
        try:
            self.video_object.BalanceWhiteAuto.set('Continuous')

        except (AttributeError, VimbaFeatureError):
            pass

        # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
        try:
            self.video_object.GVSPAdjustPacketSize.run()
            while not self.video_object.GVSPAdjustPacketSize.is_done():
                pass
        except (AttributeError, VimbaFeatureError):
            pass

        # Get properties of the camera
        self.camera_name = self.video_object.get_name()
        self.camera_model = self.video_object.get_model()
        self.camera_identifier = self.video_object.get_id()
        self.camera_serial = self.video_object.get_serial()
        self.camera_interface = self.video_object.get_interface_id()
        self.ip_address = ip_address

        # TODO - pixel formats in widget
        # px = self.video_object.get_pixel_formats()
        # print(px)
        # print(self.video_object.get_pixel_format())  # returns the current pixel format
        # set_pixel_format (fmt) # enables you to set a new pixel format

    def eof(self):
        """Overrides eof.

        Returns:
            (bool): True if video device is currently capturing.
        """
        return False

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

    def frame_count(self):
        """The number of frames in the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_COUNT property - zero if a live camera.
        """
        return 1

    def get_camera_temperature(self):
        """Queries (and returns) the temperature of the camera"""
        return self.video_object.DeviceTemperature.get()

    def exposure_time(self):
        """Queries (and returns) the current exposure time of the camera. Returned in milliseconds

        When queried, the result is in microseconds
        """
        return self.video_object.ExposureTimeAbs.get() / 1e3

    def get_frame(self):
        frame = self.video_object.get_frame()
        image = cv2.cvtColor(frame.as_numpy_ndarray(), cv2.COLOR_BAYER_RG2BGR)
        return image

    def is_available(self):
        """Returns the current status of an imaging device.
        NB: Overrides default method.

        Returns:
            (bool): True if imaging device is available.
        """
        try:
            if self.video_object.get_serial() is not None:
                return True
        except (AttributeError, VimbaFeatureError):
            return False

    def set_capture_parameters(self, configs):
        """Updates capture device properties for Vimba cameras.

        Args:
            configs (dict): dictionary of configurations.  The keys are expected to be consistent with OpenCV flags.

        Returns:
            (int): Success.
        """
        assert type(configs) is dict, "WTF!!  set_capture_parameters: A dict was expected but not received..."
        result = True

        if "CAP_PROP_FRAME_WIDTH" in configs:
            try:
                self.video_object.Width.set(int(configs["CAP_PROP_FRAME_WIDTH"]))
            except (AttributeError, VimbaFeatureError):
                pass

        if "CAP_PROP_FRAME_HEIGHT" in configs:
            try:
                self.video_object.Height.set(int(configs["CAP_PROP_FRAME_HEIGHT"]))
            except (AttributeError, VimbaFeatureError):
                pass

        if "CAP_PROP_FPS" in configs:
            try:
                self.video_object.AcquisitionFrameRateAbs.set(int(configs["CAP_PROP_FPS"]))
            except (AttributeError, VimbaFeatureError):
                pass

        return result

    def name(self):
        return '{} | {}'.format(self.camera_name, self.ip_address)

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        return "\nAllied Vision Camera Source:" \
               "\n\tVideo Device      : {0}" \
               "\n\tIP Address        : {1}" \
               "\n\tDevice Identifier : {2}" \
               "\n\tSerial Number     : {3}" \
               "\n\tInterface ID      : {4}" \
               "\n\tWidth             : {5}" \
               "\n\tHeight            : {6}" \
               "\n\tTemperature       : {7:.1f} deg C" \
               "\n\tExposure Time     : {8:.1f} ms" \
               "\n\tFrame Rate        : {9:.1f} f.p.s".format(self.camera_model,
                                                              self.ip_address,
                                                              self.camera_identifier,
                                                              self.camera_serial,
                                                              self.camera_interface,
                                                              self.width(),
                                                              self.height(),
                                                              self.get_camera_temperature(),
                                                              self.exposure_time(),
                                                              self.fps())
