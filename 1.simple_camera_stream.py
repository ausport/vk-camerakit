import cameras
import time
from tqdm import tqdm


"""
An uber-simple Vimba camera streaming example.
"""

available_vimba_devices = cameras.enumerate_vimba_devices()

if len(available_vimba_devices) == 0:
    raise IOError("No Vimba cameras are connected..")

# Load a VKCamera object.
camera = cameras.VKCameraVimbaDevice(device_id=available_vimba_devices[0].get_id(),
                                     streaming_mode=True)

# Start the device streaming to a cache.
# Note, we can get a frame at any time, but here we just wait a few seconds.
camera.start_streaming()

# Wait for a while....
time.sleep(5)

# Stop the device streaming, but the frames are still retained in cache.
camera.stop_streaming()

for _ in tqdm(range(camera.cache_size), desc=f"Capturing Frames ({camera.device_id})", ascii=True, ncols=100):
    frame = camera.get_frame()

# Other possible functions:
# camera.save_cache_to_video()
# camera.save_cache_to_images()


