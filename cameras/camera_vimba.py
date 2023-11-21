"""Camera controller for video capture from Allied Vision video camera (uses Vimba SDK)"""
import queue
import time

import cv2
import numpy as np
from vmbpy import *
from cameras import VKCamera
from cameras.helpers.vimba_utilities import set_nearest_value, get_camera, setup_pixel_format, VimbaFrameController

FEATURE_MAX = -1

VIMBA_CAPTURE_MODE_SYNCRONOUS = 0
VIMBA_CAPTURE_MODE_ASYNCRONOUS = 1


def VIMBA_INSTANCE():
    return VmbSystem.get_instance()


class VKCameraVimbaDevice(VKCamera):
    """
    See examples: https://github.com/alliedvision/VmbPy
    """

    def __init__(self, device_id,
                 streaming_mode=VIMBA_CAPTURE_MODE_SYNCRONOUS,
                 configs=None,
                 verbose_mode=False,
                 surface_name=None):

        super().__init__(surface_name=surface_name, configs=configs, verbose_mode=verbose_mode)

        print("Initialising Allied Vision device at {0}".format(device_id))

        # Asynchronous mode (VIMBA_CAPTURE_MODE_ASYNCRONOUS) or frame grab mode (VIMBA_CAPTURE_MODE_SYNCRONOUS)
        self._streaming_mode = streaming_mode
        # Device hardware address.
        self._device_id = device_id

        with VIMBA_INSTANCE():
            with get_camera(device_id) as cam:
                self.video_object = cam
                self._device_id = device_id

                # Set defaults as maximums (-1)
                self.set_capture_parameters({"CAP_PROP_FRAME_WIDTH": FEATURE_MAX,
                                             "CAP_PROP_FRAME_HEIGHT": FEATURE_MAX,
                                             "CAP_PROP_FPS": FEATURE_MAX,
                                             })

                if configs is not None:
                    # Override defaults with any custom settings
                    self.set_capture_parameters(configs)

                self._fps = float(cam.AcquisitionFrameRateAbs.get())
                self._width = int(cam.Width.get())
                self._height = int(cam.Height.get())

                setup_pixel_format(cam)

                self.update_camera_properties()
                print(self)

            # Create a frame controller that will loop
            # on a background thread, and accumulate frames to a queue.
            self._frame_queue = queue.Queue()
            self._frame_controller = VimbaFrameController(camera=self,
                                                          image_queue=self._frame_queue)
            self._streaming = False

    @property
    def device_id(self):
        return self._device_id

    def get_frame(self):
        """Returns a frame in opencv-compatible format from the Vimba device.

        If operating in VIMBA_CAPTURE_MODE_ASYNCRONOUS mode, a queued frame
        will be returned.

        If operating in VIMBA_CAPTURE_MODE_SYNCRONOUS mode, a new frame
        will be polled from the Vimba camera device.

        NB - get_frame() should be called from within a valid instance.  i.e.:

            with cameras.VIMBA_INSTANCE():
                with camera.vimba_camera() as cam:
                    while True:
                        f = camera.get_frame()

        Returns:
            Vimba frame
        """

        if self._streaming_mode == VIMBA_CAPTURE_MODE_ASYNCRONOUS:

            while True:
                # Wait for frames to be available.
                if self._frame_controller.has_queued_frames():
                    break

                # TODO - add a timeout (2 seconds??)
                # Give the streamer a chance to queue frames.
                time.sleep(0.1)

            # Get a frame from the streaming async handler.
            return self._frame_controller.next_frame()

        else:
            frame = self.video_object.get_frame()
            converted_frame = frame.convert_pixel_format(PixelFormat.Bgr8)
            opencv_image = converted_frame.as_opencv_image()

            if self.image_rotation >= 0:
                opencv_image = cv2.rotate(opencv_image, self.image_rotation)

            return opencv_image

    def frame_count(self):
        """The number of frames in the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_COUNT property - zero if a live camera.
        """
        return self.cache_size

    def width(self):
        """The pixel width of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_WIDTH property.
        """
        return self._width

    def height(self):
        """The pixel height of the video resource.

        Returns:
            (int): The CAP_PROP_FRAME_HEIGHT property.
        """
        return self._height

    def fps(self):
        """The frames per second of the video resource.

        Returns:
            (float): The CAP_PROP_FPS property.
        """
        return self._fps

    def eof(self):
        """Overrides eof.

        Returns:
            (bool): True if video device is currently capturing.
        """
        return self.cache_size == 0

    def cache_size(self):
        return self._frame_controller.cache_size()

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

    def camera_temperature(self):
        """Queries (and returns) the temperature of the camera"""
        with VIMBA_INSTANCE():
            with get_camera(self.device_id) as cam:
                return cam.DeviceTemperature.get()
        pass

    def exposure_time(self):
        """Returns the current exposure time of the device if available in milliseconds"""
        return self.video_object.ExposureTimeAbs.get() / 1e3

    def streaming_mode(self):
        return self._streaming_mode

    def streaming_mode_name(self):
        if self._streaming_mode == VIMBA_CAPTURE_MODE_SYNCRONOUS:
            return "Synchronous Capture Mode"
        elif self._streaming_mode == VIMBA_CAPTURE_MODE_ASYNCRONOUS:
            return "Asynchronous Capture Mode"
        return "N/A"

    def pre_roll(self):
        """Begin asynchronous image acquisition, but do not cache the frames (yet).

        Optional function to allow camera pre-rolling.
        Designed to allow more precise commencement of image
        caching."""
        self._frame_controller.set_pre_roll_mode(True)
        self._frame_controller.start()
        self._streaming = True

    def start_streaming(self):
        """An asynchronous image acquisition routine which will queue frames
        streaming from the connected image device."""
        if self._frame_controller.is_alive():
            self._frame_controller.set_pre_roll_mode(False)
        else:
            self._frame_controller.start()

    def stop_streaming(self):
        """Terminate threaded asynchronous image acquisition."""
        self._frame_controller.stop()
        self._streaming = False

    def is_streaming(self):
        """Verifies that the device is streaming."""
        return self._streaming

    def save_cache_to_video(self, path):
        """Dump cache to a video file"""
        try:
            video_writer = self.instantiate_writer_with_path(path)
            while self.cache_size > 0:
                cached_frame = self.get_frame()
                video_writer.write(np.asarray(cached_frame))

        except Exception as e:
            print(f"An error occurred in camera_vimba::save_cache_to_video: {e}")

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

        with VIMBA_INSTANCE():
            with self.video_object:

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

                # Set continuous white balance
                try:
                    self.video_object.BalanceWhiteAuto.set('Continuous')
                except (AttributeError, VmbFeatureError):
                    print('Camera {}: Failed to set Feature \'BalanceWhiteAuto\'.'.format(self.video_object.get_id()))

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

    # def vimba_camera(self):
    #     return self.video_object

    def name(self):
        return '{} | {}'.format(self.video_object.get_name(), self.video_object.get_id())

    def __str__(self):
        """Overriding str

        Returns:
            (str): A string summary of the object
        """
        with VIMBA_INSTANCE():
            with get_camera(self.device_id):
                return f"\nVimba-Compatible Camera Source:" \
                       f"\n\tCamera Name      : {self.video_object.get_name()}"  \
                       f"\n\tModel Name       : {self.video_object.get_model()}" \
                       f"\n\tCamera ID        : {self.video_object.get_id()}" \
                       f"\n\tInterface ID     : {self.video_object.get_interface_id()}" \
                       f"\n\tWidth            : {self.width()}" \
                       f"\n\tHeight           : {self.height()}" \
                       f"\n\tTemperature      : {self.camera_temperature()} C" \
                       f"\n\tFrame Rate       : {self.fps()} f.p.s." \
                       f"\n\tCapture Mode     : {self.streaming_mode_name()}\n"
