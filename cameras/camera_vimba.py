"""Camera controller for video capture from Allied Vision video camera (uses Vimba SDK)"""
import time

import cv2
from vmbpy import *
from cameras import VKCamera
from cameras.helpers.vimba_utilities import set_nearest_value, get_camera, setup_pixel_format, VimbaASynchronousHandler

FEATURE_MAX = -1


class VKCameraVimbaDevice(VKCamera):
    """
    See examples: https://github.com/alliedvision/VmbPy
    """

    def __init__(self, device_id, verbose_mode=False, surface_name=None, capture_path=None):
        super().__init__(surface_name=surface_name, verbose_mode=verbose_mode)

        print("Initialising Allied Vision device at {0}".format(device_id))

        with VmbSystem.get_instance():
            with get_camera(device_id) as cam:
                self.video_object = cam
                self.device_id = device_id

                # Set defaults as maximums (-1)
                self.set_capture_parameters({"CAP_PROP_FRAME_WIDTH": FEATURE_MAX,
                                             "CAP_PROP_FRAME_HEIGHT": FEATURE_MAX,
                                             "CAP_PROP_FPS": FEATURE_MAX,
                                             })
                setup_pixel_format(cam)

                _video_writer = None
                if capture_path:
                    FOURCC = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    _video_writer = cv2.VideoWriter(f"{capture_path}/capture_{device_id}.mp4", FOURCC, self.fps(), (self.width(), self.height()), True)

                self.async_handler = VimbaASynchronousHandler(camera=self, writer=_video_writer)

            print(self)

    def vimba_instance(self):
        return VmbSystem.get_instance()

    def vimba_camera(self):
        return self.video_object

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

    def camera_temperature(self):
        """Queries (and returns) the temperature of the camera"""
        with VmbSystem.get_instance():
            with get_camera(self.device_id) as cam:
                return cam.DeviceTemperature.get()
        pass

    def exposure_time(self):
        """Queries (and returns) the current exposure time of the camera. Returned in milliseconds

        When queried, the result is in microseconds
        """
        return self.video_object.ExposureTimeAbs.get() / 1e3

    def get_frame(self):
        frame = self.video_object.get_frame()
        image = cv2.cvtColor(frame.as_numpy_ndarray(), cv2.COLOR_BAYER_RG2BGR)
        return image

    def start_streaming(self):
        print(f"Spinning up streaming on device: {self.device_id}")
        with VmbSystem.get_instance():
            with get_camera(self.device_id) as cam:
                print("Connected...")
                try:
                    # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                    cam.start_streaming(handler=self.async_handler, buffer_count=10)
                    self.async_handler.shutdown_event.wait()

                finally:
                    cam.stop_streaming()

    def is_available(self):
        """Returns the current status of an imaging device.
        NB: Overrides default method.

        Returns:
            (bool): True if imaging device is available.
        """
        try:
            if self.video_object.get_serial() is not None:
                return True
        except (AttributeError, VmbFeatureError):
            return False

    def set_capture_parameters(self, configs: dict):
        """Updates capture device properties for Vimba cameras.
        The default instance of this method assumes the capture device is OpenCV-compatible.
        This method should be overridden for other devices (e.g. Vimba-compatible IP cameras).

        Args:
            configs (dict): dictionary of configurations.  The keys are expected to be consistent with OpenCV flags.

        Returns:
            (int): Success.
        """
        assert type(configs) is dict, "WTF!!  set_capture_parameters: A dict was expected but not received..."
        result = True

        # Set continuous exposure
        try:
            self.video_object.ExposureAuto.set('Continuous')
        except (AttributeError, VmbFeatureError):
            print('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(self.video_object.get_id()))

        # Set continuous gain
        try:
            self.video_object.GainAuto.set('Continuous')
        except (AttributeError, VmbFeatureError):
            print('Camera {}: Failed to set Feature \'GainAuto\'.'.format(self.video_object.get_id()))

        try:
            stream = self.video_object.get_streams()[0]
            stream.GVSPAdjustPacketSize.run()
            while not stream.GVSPAdjustPacketSize.is_done():
                pass
        except (AttributeError, VmbFeatureError):
            print('Camera {}: Failed to set Feature \'GVSPAdjustPacketSize\'.'.format(self.video_object.get_id()))

        if "CAP_PROP_FRAME_WIDTH" in configs:
            try:
                set_nearest_value(self.video_object, 'Width', int(configs["CAP_PROP_FRAME_WIDTH"]))
            except (AttributeError, VmbFeatureError):
                pass

        if "CAP_PROP_FRAME_HEIGHT" in configs:
            try:
                set_nearest_value(self.video_object, 'Height', int(configs["CAP_PROP_FRAME_HEIGHT"]))
            except (AttributeError, VmbFeatureError):
                pass

        if "CAP_PROP_FPS" in configs:
            try:
                set_nearest_value(self.video_object, 'AcquisitionFrameRateAbs', int(configs["CAP_PROP_FPS"]))
            except (AttributeError, VmbFeatureError):
                pass

        return result

    def name(self):
        return '{} | {}'.format(self.video_object.get_name(), self.video_object.get_id())

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        with VmbSystem.get_instance():
            with get_camera(self.device_id):
                return f"\nEthernet Camera Source:" \
                       f"\n\tCamera Name      : {self.video_object.get_name()}"  \
                       f"\n\tModel Name       : {self.video_object.get_model()}" \
                       f"\n\tCamera ID        : {self.video_object.get_id()}" \
                       f"\n\tInterface ID     : {self.video_object.get_interface_id()}" \
                       f"\n\tWidth            : {self.width()}" \
                       f"\n\tHeight           : {self.height()}" \
                       f"\n\tTemperature      : {self.camera_temperature()} C" \
                       f"\n\tFrame Rate       : {self.fps()} f.p.s."
