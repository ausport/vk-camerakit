import cameras
import time
import os
from tqdm import tqdm
from vmbpy import *

"""
An uber-simple Vimba camera streaming example.
"""

available_vimba_devices = cameras.enumerate_vimba_devices()

if len(available_vimba_devices) == 0:
    raise IOError("No Vimba cameras are connected..")

my_cameras = []
for device in available_vimba_devices:
    # Load a VKCamera object.
    camera = cameras.VKCameraVimbaDevice(device_id=device.get_id(),
                                         configs={"CAP_PROP_FPS": 50},
                                         streaming_mode=True)

    my_cameras.append(camera)

"""Note that multiple cameras require a with vimba instance statement block"""
with cameras.VIMBA_INSTANCE():
    # Start the device streaming to a cache.
    # Note, we can get a frame at any time, but here we just wait a few seconds.
    for camera in my_cameras:
        print(f"Starting {camera.device_id}")
        camera.start_streaming()

    # Wait for a while....
    time.sleep(10)

    # Stop the device streaming, but the frames are still retained in cache.
    for camera in my_cameras:
        camera.stop_streaming()

    for camera in my_cameras:
        file_path = os.path.join(os.getcwd(), f"CAPTURE_{camera.device_id}.mp4")
        print(f"Writing {camera.cache_size} frames to {file_path}")
        camera.save_cache_to_video(path=file_path)



