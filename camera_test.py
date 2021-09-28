import cameras


def test_class(obj):
    print(obj)
    print(obj.surface_model)
    img = obj.get_frame()
    print(img.shape)
    obj.close()


# Test case 1 - load video file with surface.
test_class(cameras.VKCameraVideoFile(filepath="/Users/stuartmorgan2/Dropbox/_Microwork/Multiview/2_view/Hockey/Hockey_2_A.mp4",
                                     surface_name="Hockey"))

# Test case 1.1 - load video file with no surface.
test_class(cameras.VKCameraVideoFile(filepath="/Users/stuartmorgan2/Dropbox/_Microwork/Multiview/2_view/Hockey/Hockey_2_A.mp4"))

# Test case 2 - load video device with surface.
test_class(cameras.VKCameraGenericDevice(device=0,
                                         surface_name="Tennis"))

# Test case 3 - load config file
m = cameras.load_camera_model_from_json(path="/Users/stuartmorgan2/Desktop/extrinsics_sample_tennis.json")
test_class(m)
