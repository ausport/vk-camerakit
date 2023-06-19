import cameras
import os

# Check for Vimba cameras
vimba_cameras = cameras.enumerate_vimba_devices()
choice = 0
CAPTURE_PATH = "./capture"

if not os.path.exists(CAPTURE_PATH):
    os.mkdir(CAPTURE_PATH)

if len(vimba_cameras) > 1:
    available_devices = []

    for camera in vimba_cameras:
        # Add vimba camera object to VKCamera wrapper.
        camera_model = cameras.VKCameraVimbaDevice(device_id=camera.get_id())
        print("Vimba-Compatible Camera Found:", camera.__class__)

        if camera_model.is_available():
            available_devices.append(camera.get_id())

    print("Select an option:")
    for i, option in enumerate(available_devices):
        print(f"{i}. {option}")

    while True:
        choice = input("Enter your choice (0-{0}): ".format(len(available_devices)-1))

        if choice == "":
            choice = 0
            break

        if not choice.isdigit() or int(choice) < 0 or int(choice) > len(available_devices):
            print("Invalid choice. Please try again.")
        else:
            break

elif len(vimba_cameras) == 0:
    exit(1)

camera = cameras.VKCameraVimbaDevice(device_id=vimba_cameras[int(choice)].get_id(), capture_path=CAPTURE_PATH)

camera.set_image_rotation(cameras.VK_ROTATE_180)

# NB- Vimba camera capture calls need to exist in a Vimba context.
# TODO - put this on a thread..
with camera.vimba_instance():
    with camera.vimba_camera() as cam:
        while True:
            f = camera.get_frame()

'''
    with VmbSystem.get_instance():
        with get_camera(cam_id) as cam:
            setup_camera(cam)

            _video_writer = cv2.VideoWriter(f"{CAPTURE_PATH}/capture_{cam.get_id()}.mp4", FOURCC, 25, (1456, 1088), True)

            start_time = time.time()
            # Acquire 10 frame with a custom timeout (default is 2000ms) per frame acquisition.
            while True:
                for frame in cam.get_frame_generator(limit=None, timeout_ms=100):
                    print('Got {}'.format(frame), flush=True)

                    display = frame.convert_pixel_format(PixelFormat.Bgr8)
                    _video_writer.write(display.as_numpy_ndarray())

            # End the timer
            end_time = time.time()

            # Calculate the execution time
            execution_time = end_time - start_time

            # Print the execution time
            t = (execution_time*1000.)/FRAMES
            print(f"The for-loop took {execution_time} seconds to execute at {t} f.p.s.")
'''