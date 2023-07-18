"""Camera controller for video capture from Allied Vision video camera (uses Vimba SDK)"""
import time
import threading
import numpy as np

import cv2
from vmbpy import *
import cameras
from cameras import VKCamera
from cameras.helpers.vimba_utilities import set_nearest_value, get_camera, setup_pixel_format, VimbaASynchronousHandler

FEATURE_MAX = -1


class VKCameraVimbaDevice(VKCamera):
    """
    See examples: https://github.com/alliedvision/VmbPy
    """

    def __init__(self, device_id, verbose_mode=False, surface_name=None):
        super().__init__(surface_name=surface_name, verbose_mode=verbose_mode)

        print("Initialising Allied Vision device at {0}".format(device_id))

        self.shutdown_event = threading.Event()

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

                self.async_handler = VimbaASynchronousHandler(camera=self)

                self.update_camera_properties()
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
        """
        Returns a frame in opencv-compatible format from the Vimba device.
        NB - get_frame() should be called from within a valid instance.  i.e.:

            with camera.vimba_instance():
                with camera.vimba_camera() as cam:
                    while True:
                        f = camera.get_frame()

        Returns:
            Vimba frame
        """
        frame = self.video_object.get_frame()
        converted_frame = frame.convert_pixel_format(PixelFormat.Bgr8)
        opencv_image = converted_frame.as_opencv_image()

        if self.image_rotation >= 0:
            opencv_image = cv2.rotate(opencv_image, self.image_rotation)

        return opencv_image

    def generate_frames(self, vimba_context, path=None, limit=None, show_frames=False):

        _video_writer = None
        if path is not None:
            _video_writer = self.instantiate_writer_with_path(path=path)

        start_time = time.time()
        loop_counter = 0
        ENTER_KEY_CODE = 13

        while True:
            for frame in vimba_context.get_frame_generator(limit=limit, timeout_ms=2000):
                loop_counter += 1

                converted_frame = frame.convert_pixel_format(PixelFormat.Bgr8)
                opencv_image = converted_frame.as_opencv_image()

                if self.image_rotation is not cameras.VK_ROTATE_NONE:
                    opencv_image = cv2.rotate(opencv_image, self.image_rotation)

                if _video_writer:
                    _video_writer.write(np.asarray(opencv_image))

                if show_frames:
                    key = cv2.waitKey(1)
                    if key == ENTER_KEY_CODE:
                        self.shutdown_event.set()
                        return

                    msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
                    cv2.imshow(msg.format(vimba_context.get_name()), opencv_image)

                # Check if one second has passed
                if time.time() - start_time >= 1:
                    print("Frames per second in the last one-second interval: {}".format(loop_counter))
                    loop_counter = 0
                    start_time = time.time()

    def start_streaming(self, vimba_context, path=None, limit=None, show_frames=False):
        print(f"Spinning up streaming on device: {self.device_id}")

        # Set updated handler properties
        if path:
            self.async_handler.set_video_writer(video_writer=self.instantiate_writer_with_path(path=path))

        self.async_handler.set_show_frames(show_frames)

        try:
            # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
            vimba_context.start_streaming(handler=self.async_handler, buffer_count=10)
            self.async_handler.shutdown_event.wait()

        finally:
            vimba_context.stop_streaming()

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
                # self.video_object.TriggerSource.set('FreeRun')
                self.video_object.TriggerSource.set('FixedRate')
                set_nearest_value(self.video_object, 'AcquisitionFrameRateAbs', int(configs["CAP_PROP_FPS"]))
            except (AttributeError, VmbFeatureError):
                pass

        if "CAP_PROP_ROTATION" in configs:
            self.set_image_rotation(int(configs["CAP_PROP_ROTATION"]))

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
