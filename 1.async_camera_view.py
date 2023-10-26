import time
import threading
import cameras
import os
import math
from datetime import datetime

import argparse
import os.path


def parse_args():
    parser = argparse.ArgumentParser(description='Command-line argument parser')
    parser.add_argument('-v', '--view', action='store_true', help='Enable viewing')
    parser.add_argument('-f', '--flip', action='store_true', help='Flip viewing')
    parser.add_argument('-c', '--camera_id', default=None, help='Camera ID (optional)')
    parser.add_argument('-l', '--limit', type=int, default=math.inf, help='Capture time in seconds (optional)')
    parser.add_argument('-r', '--fps', type=int, default=50, help='Frame rate (optional)')
    parser.add_argument('-d', '--destination', default=None, help='Destination directory (optional)')
    parser.add_argument('-g', '--config', default=None, help='Camera configuration file (optional)')
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
    fps = args.fps
    config = args.config

    # Interpret tilde in the destination path, if provided
    if destination:
        destination = expand_tilde(destination)
        if not os.path.exists(destination):
            os.mkdir(destination)

    # Print the parsed arguments
    print(f'Camera ID: {camera_id}')
    print(f'View enabled: {enable_view}')
    print(f'Capture time: {limit} seconds')
    print(f'Destination: {destination}')
    print(f'Flip: {flip}')
    print(f'FPS: {fps}')
    print(f'Config: {config}')

    # Find camera id if selected, otherwise enumerate and user-choice.
    vimba_cameras = []
    choice = 0

    if config is not None:
        # Load camera by config file (Ignore camera id).
        camera = cameras.load_camera_model(path=config)
        device_id = camera.device_id
    else:

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

        device_id = vimba_cameras[int(choice)].get_id()

        # Initialise specified device
        camera = cameras.VKCameraVimbaDevice(device_id=device_id)

    if destination is not None:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%y%m%d%H%M%S")
        destination = os.path.join(destination, f"capture_{device_id}_{formatted_datetime}.mp4")

    # NB- Vimba camera capture calls need to exist in a Vimba context.
    with cameras.VIMBA_INSTANCE():
        with camera.vimba_camera() as cam:

            camera.set_capture_parameters(configs={"CAP_PROP_FRAME_WIDTH": 1456,
                                                   "CAP_PROP_FRAME_HEIGHT": 1088,
                                                   "CAP_PROP_FPS": fps,
                                                   })

            if flip:
                camera.set_capture_parameters(configs={"CAP_PROP_ROTATION": cameras.VK_ROTATE_180})

            # camera.start_streaming(vimba_context=cam,
            #                        path=destination,
            #                        limit=limit,
            #                        show_frames=enable_view)

            # Create a thread to run the streaming function
            streaming_thread = threading.Thread(target=camera.start_streaming,
                                                args=(cam,
                                                destination,
                                                enable_view))

            streaming_thread.start()

            n = 0
            while True:
                print("Doing something else...")
                time.sleep(1)
                n += 1
                if n > limit:
                    camera.stop_streaming()
                    streaming_thread.join()
                    break

            print("Streaming thread has finished.")

if __name__ == '__main__':
    main()
