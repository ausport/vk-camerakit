import cameras
import os
import time
import cv2
import threading
import numpy as np

import argparse
import os.path


def parse_args():
    parser = argparse.ArgumentParser(description='Command-line argument parser')
    parser.add_argument('-v', '--view', action='store_true', help='Enable viewing')
    parser.add_argument('-f', '--flip', action='store_true', help='Flip viewing')
    parser.add_argument('-c', '--camera_id', default=None, help='Camera ID (optional)')
    parser.add_argument('-l', '--limit', type=int, default=None, help='Limit integer (optional)')
    parser.add_argument('-d', '--destination', default=None, help='Destination path (optional)')
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

    shutdown_event = threading.Event()

    # Interpret tilde in the destination path, if provided
    if destination:
        destination = expand_tilde(destination)
        if not os.path.exists(destination):
            os.mkdir(destination)

    # Print the parsed arguments
    print(f'Camera ID: {camera_id}')
    print(f'View enabled: {enable_view}')
    print(f'Limit: {limit}')
    print(f'Destination: {destination}')
    print(f'Flip: {flip}')

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

    camera = cameras.VKCameraVimbaDevice(device_id=vimba_cameras[int(choice)].get_id())

    if flip:
        camera.set_image_rotation(cameras.VK_ROTATE_180)

    _destination_width = 1456
    _destination_height = 1088
    _destination_fps = 25

    # NB- Vimba camera capture calls need to exist in a Vimba context.
    with camera.vimba_instance():
        with camera.vimba_camera() as vimba_device:
            if destination:
                FOURCC = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                _video_writer = cv2.VideoWriter(f"{destination}/capture_{vimba_device.get_id()}.mp4",
                                                FOURCC,
                                                _destination_fps,
                                                (_destination_width, _destination_height),
                                                True)

            start_time = time.time()
            loop_counter = 0
            ENTER_KEY_CODE = 13

            while True:
                loop_counter += 1
                opencv_image = camera.get_frame()

                if destination:
                    # NB: The converted frame is a Vimba frame object, not a cv-compliant numpy array
                    # Convert to ndarray to write to file.
                    _video_writer.write(np.asarray(opencv_image))
                    pass

                if enable_view:
                    key = cv2.waitKey(1)
                    if key == ENTER_KEY_CODE:
                        shutdown_event.set()
                        return

                    msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
                    cv2.imshow(msg.format(vimba_device.get_name()), opencv_image)

                # Check if one second has passed
                if time.time() - start_time >= 1:
                    print("Frames per second in the last one-second interval: {}".format(loop_counter))
                    loop_counter = 0
                    start_time = time.time()

if __name__ == '__main__':
    main()