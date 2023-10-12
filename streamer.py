import cameras
import time
import threading
from vmbpy import *

all_cameras = [
    cameras.VKCameraVimbaDevice(device_id="DEV_000F315DE931", streaming_mode=True),
    cameras.VKCameraVimbaDevice(device_id="DEV_000F3102321D", streaming_mode=True),
]

# Create and start threads for each camera
camera_threads = []


with VmbSystem.get_instance():
    for camera in all_cameras:
        print(camera.device_id ,"-->", camera.video_object.get_interface_id())
        with camera.vimba_camera() as vimba_device:

            camera.set_capture_parameters(configs={"CAP_PROP_FRAME_WIDTH": 1456,
                                                   "CAP_PROP_FRAME_HEIGHT": 1088,
                                                   "CAP_PROP_FPS": 50,
                                                   })

            streaming_thread = None

            # Create a non-blocking thread to run the streaming function
            if camera.streaming_mode() == cameras.VIMBA_CAPTURE_MODE_ASYNCRONOUS:
                streaming_thread = threading.Thread(target=camera.start_streaming,
                                                    args=(vimba_device,
                                                          None,
                                                          False))

                camera_threads.append(streaming_thread)
                streaming_thread.start()
                time.sleep(0.1)

    while True:
        time.sleep(1)

        for camera in all_cameras:
            with camera.vimba_camera() as vimba_device:
                print(camera.device_id, "-->", camera.cache_size)


#             camera.stop_streaming()
#             streaming_thread.join()
#
# print("\nHere...")
#
# exit(1)
