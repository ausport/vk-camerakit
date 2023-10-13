import multiprocessing
import time

import cameras
import threading
from multiprocessing import Process, Queue
from vmbpy import *

all_cameras = [
    cameras.VKCameraVimbaDevice(device_id="DEV_000F315DE930", streaming_mode=True),
    cameras.VKCameraVimbaDevice(device_id="DEV_000F315DE932", streaming_mode=True),
]


def stream(camera, frame_queue):
    with VmbSystem.get_instance():
        print(camera.device_id ,"-->", camera.video_object.get_interface_id())
        with camera.vimba_camera() as vimba_device:

            camera.set_capture_parameters(configs={"CAP_PROP_FRAME_WIDTH": 1456,
                                                   "CAP_PROP_FRAME_HEIGHT": 1088,
                                                   "CAP_PROP_FPS": 25,
                                                   })

            # Create a non-blocking thread to run the streaming function
            streaming_thread = threading.Thread(target=camera.start_streaming,
                                                args=(vimba_device,
                                                      None,
                                                      False))

            if camera.device_id == "DEV_000F315DE930":
                # Let one of the cameras lag behind...
                time.sleep(5)

            streaming_thread.start()

            while camera.cache_size < 200:
                if camera.cache_size % 20 == 0:
                    print(f"{camera.device_id} has queued {camera.cache_size} frames..")
                    time.sleep(0.5)

            camera.stop_streaming()
            streaming_thread.join()
            print(f"{camera.device_id} finished with {camera.cache_size} frames..")

            while camera.cache_size > 0:
                frame = camera.get_frame()
                if camera.cache_size % 20 == 0:
                    print(f"{camera.device_id} still has {camera.cache_size} frames..")
                frame_queue.put(frame)

            print(f"{camera.device_id} has completed..")


if __name__ == '__main__':
    frame_queue = Queue()

    process1 = Process(target=stream, args=(all_cameras[0], frame_queue))
    process2 = Process(target=stream, args=(all_cameras[1], frame_queue))

    process1.start()
    process2.start()

    while True:
        time.sleep(10)
        print("Killing 1")
        process1.kill()
        print("Killing 2")
        process2.kill()
        break


