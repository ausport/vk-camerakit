from vmbpy import *
from typing import Optional
import cv2
import numpy as np
import threading
import time
import queue

import cameras
from cameras import VKCamera

opencv_display_format = PixelFormat.Bgr8

FEATURE_MAX = -1


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


class VimbaASynchronousHandler:
    def __init__(self, camera: VKCamera):
        self.shutdown_event = threading.Event()
        self._writer = None
        self._show_frames = False
        self._parent_camera = camera
        self._start_time = time.time()
        self._writer_thread = VimbaVideoWriterThread(video_writer=None)

    def set_video_writer(self, video_writer):
        self._writer = video_writer
        self._writer_thread = VimbaVideoWriterThread(video_writer=video_writer)

    def set_show_frames(self, show_frames):
        self._show_frames = show_frames

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        ENTER_KEY_CODE = 13

        key = cv2.waitKey(1)
        if key == ENTER_KEY_CODE:
            # self._writer_thread.stop()
            self.shutdown_event.set()
            return

        elif frame.get_status() == FrameStatus.Complete:
            # Convert frame if it is not already the correct format
            converted_frame = frame.convert_pixel_format(PixelFormat.Bgr8)
            opencv_image = converted_frame.as_opencv_image()

            # Undistort
            opencv_image = self._parent_camera.undistorted_image(opencv_image)

            if self._parent_camera.image_rotation is not cameras.VK_ROTATE_NONE:
                opencv_image = cv2.rotate(opencv_image, self._parent_camera.image_rotation)

            # TODO - shift the writing to a background writer thread + tracking/ML??
            self._writer_thread(frame=opencv_image)
            # if self._writer:
            #     self._writer.write(np.asarray(opencv_image))

            if self._show_frames:
                key = cv2.waitKey(1)
                if key == ENTER_KEY_CODE:
                    self.shutdown_event.set()
                    return

                msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
                cv2.imshow(msg.format(cam.get_id()), opencv_image)

        cam.queue_frame(frame)

        elapsed_time = time.time() - self._start_time
        milliseconds = elapsed_time * 1000
        operations_per_second = 1 / elapsed_time

        print('{} acquired {} in {} - {} fps'.format(cam, frame, milliseconds, operations_per_second), flush=True)
        available_cache = self._writer_thread.available_cache
        print(f"Available cache: {available_cache}")

        self._start_time = time.time()


class VimbaVideoWriterThread:
    def __init__(self, video_writer):
        self.video_writer = video_writer
        self.frame_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._write_frames)
        self.thread.start()

    def __call__(self, frame):
        if not self.stop_event.is_set():
            print("Putting a frame..")
            self.frame_queue.put(frame)

    def _write_frames(self):
        time.sleep(0.001)
        # while not self.stop_event.is_set() or not self.frame_queue.empty():
        #     if not self.frame_queue.empty():
        #         frame = self.frame_queue.get()
        #         self.video_writer.write(frame)
        #         print("Writing")
        #     else:
        #         # Sleep briefly to avoid excessive CPU usage
        #         time.sleep(0.001)

    @property
    def available_cache(self):
        return self.frame_queue.maxsize - self.frame_queue.qsize()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.video_writer.release()
