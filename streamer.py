import cameras
import time
import threading
from multiprocessing import Process
from vmbpy import *

all_cameras = [
    cameras.VKCameraVimbaDevice(device_id="DEV_000F315DE930", streaming_mode=True),
    cameras.VKCameraVimbaDevice(device_id="DEV_000F315DE932", streaming_mode=True),
]

# Create and start threads for each camera
camera_threads = []


def stream(camera):
    with VmbSystem.get_instance():
        print(camera.device_id ,"-->", camera.video_object.get_interface_id())
        with camera.vimba_camera() as vimba_device:

            camera.set_capture_parameters(configs={"CAP_PROP_FRAME_WIDTH": 1456,
                                                   "CAP_PROP_FRAME_HEIGHT": 1088,
                                                   "CAP_PROP_FPS": 25,
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
                time.sleep(5)
                print(camera.device_id, "-->", camera.cache_size)
                camera.stop_streaming()
                streaming_thread.join()


process1 = Process(target=stream, args=(all_cameras[0],))
process2 = Process(target=stream, args=(all_cameras[1],))

process1.start()
process2.start()

# Wait for processes to finish
process1.join()
process2.join()

