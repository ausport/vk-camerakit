import cameras
import time
from tqdm import tqdm


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
                                         configs={"CAP_PROP_FPS": 25},
                                         streaming_mode=True)

    my_cameras.append(camera)

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
    print(f"{camera.device_id}: We are done...{camera.cache_size}")

# for _ in tqdm(range(camera.cache_size()), desc=f"Capturing Frames ({camera.device_id})", ascii=True, ncols=100):
#     frame = camera.get_frame()

# Other possible functions:
# camera.save_cache_to_video()
# camera.save_cache_to_images()


