import cameras

camera_model = cameras.VKCameraGenericDevice(device=0)

if camera_model.is_available():
    print("Generic Camera Found:", camera_model.__class__)
    print(camera_model)

# Check for Vimba cameras
vimba_cameras = cameras.enumerate_vimba_devices()

for camera in vimba_cameras:
    # Add vimba camera object to VKCamera wrapper.
    print("Generic Camera Found:", camera_model.__class__)
    camera_model = cameras.VKCameraVimbaDevice(device_id=camera.get_id())
    if camera_model.is_available():
        # print(camera_model)
        print("Camera is running...")