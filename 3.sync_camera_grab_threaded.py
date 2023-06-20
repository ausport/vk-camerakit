import cameras
import os

import argparse
import os.path

CAPTURE_PATH = "./capture"

def parse_args():
    parser = argparse.ArgumentParser(description='Command-line argument parser')
    parser.add_argument('-v', '--view', action='store_true', help='Enable viewing')
    parser.add_argument('-f', '--flip', action='store_true', help='Flip viewing')
    parser.add_argument('-c', '--camera_id', default=None, help='Camera ID (optional)')
    parser.add_argument('-l', '--limit', type=int, default=None, help='Limit integer (optional)')
    parser.add_argument('-d', '--destination', default=CAPTURE_PATH, help='Destination path (optional)')
    return parser.parse_args()


def expand_tilde(path):
    return os.path.expanduser(path)


def main():
    args = parse_args()
    camera_id = args.camera_id
    enable_view = args.view
    limit = args.limit
    destination = args.destination
    flip = args.flip

    # Interpret tilde in the destination path, if provided
    if destination:
        destination = expand_tilde(destination)

    # Print the parsed arguments
    print(f'Camera ID: {camera_id}')
    print(f'View enabled: {enable_view}')
    print(f'Limit: {limit}')
    print(f'Destination: {destination}')
    print(f'Flip: {flip}')

    if not os.path.exists(destination):
        os.mkdir(destination)

    # Find camera id if selected, otherwise enumerate and user-choice.
    vimba_cameras = []
    choice = 0

    if camera_id is not None:
        vimba_cameras.append(cameras.get_camera(camera_id))

    else:
        # Check for all Vimba cameras
        available_device_ids = []
        vimba_cameras = cameras.enumerate_vimba_devices()

        if len(vimba_cameras) == 0:
            print("No Vimba-compatible devices were found.")
            exit(1)

        else:
            for camera in vimba_cameras:
                # Add vimba camera object to VKCamera wrapper.
                camera_model = cameras.VKCameraVimbaDevice(device_id=camera.get_id())
                print("Vimba-Compatible Camera Found:", camera.__class__)

                if camera_model.is_available():
                    available_device_ids.append(camera.get_id())

            print("Select an option:")
            for i, option in enumerate(available_device_ids):
                print(f"{i}. {option}")

            while True:
                choice = input("Enter your choice (0-{0}): ".format(len(available_device_ids)-1))

                if choice == "":
                    choice = 0
                    break

                if not choice.isdigit() or int(choice) < 0 or int(choice) > len(available_device_ids):
                    print("Invalid choice. Please try again.")
                else:
                    break

    camera = cameras.VKCameraVimbaDevice(device_id=vimba_cameras[int(choice)].get_id(), capture_path=CAPTURE_PATH)

    if flip:
        camera.set_image_rotation(cameras.VK_ROTATE_180)

    _destination_width = 1456
    _destination_height = 1088
    _destination_fps = 25

    # NB- Vimba camera capture calls need to exist in a Vimba context.
    with camera.vimba_instance():
        with camera.vimba_camera() as cam:

            camera.generate_frames(vimba_device=cam,
                                   w=_destination_width, h=_destination_height,
                                   path=destination,
                                   fps=_destination_fps, limit=limit,
                                   show_frames=True)


if __name__ == '__main__':
    main()
