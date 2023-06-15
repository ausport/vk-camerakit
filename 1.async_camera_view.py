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

print(camera.camera_temperature())
camera.start_streaming()
