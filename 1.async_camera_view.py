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

camera = cameras.VKCameraVimbaDevice(device_id=vimba_cameras[int(choice)].get_id())

# NB- Vimba camera capture calls need to exist in a Vimba context.
with camera.vimba_instance():
    with camera.vimba_camera() as cam:

        camera.set_capture_parameters(configs={"CAP_PROP_FRAME_WIDTH": 1480,
                                               "CAP_PROP_FRAME_HEIGHT": 1088,
                                               "CAP_PROP_FPS": 25,
                                               "CAP_PROP_ROTATION": cameras.VK_ROTATE_180,
                                               })

        camera.start_streaming(vimba_device=cam)
