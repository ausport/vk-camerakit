from vmbpy import *
from typing import Optional
import cv2
import threading
import queue

import cameras
from cameras import VKCamera


def enumerate_vimba_devices():

    with VmbSystem.get_instance () as vmb:
        cams = vmb.get_all_cameras()
        print(f'\n{len(cams)} Vimba camera(s) found...')
    return cams


def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)

            except VmbCameraError:
                print('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vmb.get_all_cameras()
            if not cams:
                print('No Cameras accessible. Abort.')

            return cams[0]


def setup_pixel_format(cam: Camera):
    # Pixel format
    opencv_display_format = PixelFormat.Bgr8

    # Query available pixel formats. Prefer color formats over monochrome formats
    cam_formats = cam.get_pixel_formats()
    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color_formats = tuple(f for f in cam_color_formats
                                      if opencv_display_format in f.get_convertible_formats())

    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono_formats = tuple(f for f in cam_mono_formats
                                     if opencv_display_format in f.get_convertible_formats())

    # if OpenCV compatible color format is supported directly, use that
    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)

    # else if existing color format can be converted to OpenCV format do that
    elif convertible_color_formats:
        cam.set_pixel_format(convertible_color_formats[0])

    # fall back to a mono format that can be converted
    elif convertible_mono_formats:
        cam.set_pixel_format(convertible_mono_formats[0])

    else:
        print('Camera does not support an OpenCV compatible format. Abort.')


def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
    # Helper function that tries to set a given value. If setting of the initial value failed
    # it calculates the nearest valid value and sets the result. This function is intended to
    # be used with Height and Width Features because not all Cameras allow the same values
    # for height and width.
    FEATURE_MAX = -1
    with cam:
        feat = cam.get_feature_by_name(feat_name)

        min_, max_ = feat.get_range()
        # print(f"{feat_name} - range: {min_} to {max_}")

        if feat_value == FEATURE_MAX:
            feat_value = max_

        try:
            feat.set(feat_value)

        except VmbFeatureError:
            min_, max_ = feat.get_range()
            # print(f"{feat_name} - range: {min_} to {max_}")
            inc = feat.get_increment()

            if feat_value <= min_:
                val = min_

            elif feat_value >= max_:
                val = max_

            else:
                val = (((feat_value - min_) // inc) * inc) + min_

            feat.set(val)

            msg = ('Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
                   'Using nearest valid value \'{}\'. Note that, this causes resizing '
                   'during processing, reducing the frame rate.')
            Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))


class VimbaASynchronousStreamHandler:
    def __init__(self, camera: VKCamera):
        self.shutdown_event = threading.Event()
        self._parent_camera = camera
        self._frame_handler = VimbaASynchronousFrameHandler(parent_async_handler=self)

    def has_queued_frames(self):
        return self._frame_handler.cache_size > 0

    @property
    def cache_size(self):
        return self._frame_handler.cache_size

    def is_streaming(self):
        return self._parent_camera.video_object.is_streaming()

    def next_frame(self):
        return self._frame_handler.next_frame_from_queue()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            # We convert the image to opencv format here, and only once.
            converted_frame = frame.convert_pixel_format(PixelFormat.Bgr8)
            undistorted_opencv_image = self._parent_camera.undistorted_image(converted_frame.as_opencv_image())

            if self._parent_camera.image_rotation is not cameras.VK_ROTATE_NONE:
                undistorted_opencv_image = cv2.rotate(undistorted_opencv_image, self._parent_camera.image_rotation)

            # Queue the frame with the frame handler
            self._frame_handler(frame=undistorted_opencv_image)

        cam.queue_frame(frame)


class VimbaASynchronousFrameHandler:
    def __init__(self, parent_async_handler):
        self.parent = parent_async_handler
        self.frame_queue = queue.Queue()
        self.shutdown_event = threading.Event()

    def __call__(self, frame):
        if not self.shutdown_event.is_set():
            self.frame_queue.put(frame)

    def next_frame_from_queue(self):
        """
        Return a valid opencv frame from the queue.
        """
        if self.frame_queue.empty():
            return None
        else:
            return self.frame_queue.get()

    @property
    def cache_size(self):
        return self.frame_queue.qsize()

