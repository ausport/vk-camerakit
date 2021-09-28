import json
import os
import numpy as np
from cameras import VKCameraVideoFile, VKCameraPanorama, VKCameraGenericDevice


def load_camera_model_from_json(path):

    assert os.path.exists(path), "JSON (camera) path does not exist: {0}".format(path)

    with open(path) as data_file:
        _data = json.load(data_file)

        # Load the surface model first
        if "surface_model" in _data:
            _surface_model_name = _data["surface_model"]
        else:
            _surface_model_name = None

        # Define the camera class mode.
        _vk_camera_class = "VKCamera"
        if "class" in _data:
            _vk_camera_class = _data["class"]

        if _vk_camera_class == "VKCamera":
            # Attempt to initiate a camera device..
            camera_model = VKCameraGenericDevice(device=0, surface_name=_surface_model_name)

        elif _vk_camera_class == "VKCameraVideoFile":
            assert "image_path" in _data, "Camera file doesn't include an image path..."
            camera_model = VKCameraVideoFile(filepath=_data["image_path"], surface_name=_surface_model_name)

        # elif _vk_camera_class == "VKCameraPanorama":
        #     assert "image_paths" in j, "Camera file doesn't include an image path..."
        #     # Load a single video file.
        #     # Load multiple video files as panorama
        #     self.enable_panorama_mode()

        # Load additional parameters if available
        if camera_model.surface_model is not None:
            if "homography" in _data:
                print("Loading homography")
                camera_model.surface_model.homography = np.asarray(_data["homography"])
                camera_model.surface_model.compute_inverse_homography()

            if "image_points" in _data:
                camera_model.surface_model.image_points = np.asarray(_data["image_points"])

            if "model_points" in _data:
                camera_model.surface_model.model_points = np.asarray(_data["model_points"])

        if "distortion_matrix" in _data:
            camera_model.distortion_matrix = np.asarray(_data["distortion_matrix"])

        if "camera_matrix" in _data:
            camera_model.camera_matrix = np.asarray(_data["camera_matrix"])

        if "focal_length" in _data:
            camera_model.focal_length = _data["focal_length"]

    return camera_model

