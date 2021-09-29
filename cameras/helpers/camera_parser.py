import json
import os
import numpy as np
from cameras import VKCameraVideoFile, VKCameraPanorama, VKCameraGenericDevice, VKCamera


def parse_camera_model_with_dict(data):
    """Extract camera model parameters from dict object.

    Args:
        data (dict): dict object, derived from json file or directly from a UI.

    Returns:
        camera_model (VKCamera): model representation camera described by the data object.
    """

    camera_model = VKCamera()

    # Load the surface model first
    if "surface_model" in data:
        _surface_model_name = data["surface_model"]
    else:
        _surface_model_name = None

    # Define the camera class mode.
    _vk_camera_class = "VKCamera"

    if "class" in data:
        _vk_camera_class = data["class"]

    if _vk_camera_class == "VKCamera":
        # Attempt to initiate a camera device..
        camera_model = VKCameraGenericDevice(device=0, surface_name=_surface_model_name)

    elif _vk_camera_class == "VKCameraVideoFile":
        assert "image_path" in data, "Camera file doesn't include an image path..."
        camera_model = VKCameraVideoFile(filepath=data["image_path"], surface_name=_surface_model_name)

    elif _vk_camera_class == "VKCameraPanorama":
        assert "stitching_parameters" in data, "Camera file doesn't include stitching parameters..."
        assert "panorama_projection_models" in data, "Camera file doesn't include panorama projection parameters..."

        input_cameras = []
        panorama_projection_models = []

        for camera in data["panorama_projection_models"]:

            # Parse an initialised camera model from the data
            input_cameras.append(parse_camera_model_with_dict(camera["input_camera_model"]))

            # Ensure matrices are in np-compliant form.
            for part in camera["projection_model_parameters"]:
                if part == "extrinsics" or part == "rotation":
                    camera["projection_model_parameters"][part] = np.asarray(camera["projection_model_parameters"][part])

            # Retrieve the panoramic model dict
            panorama_projection_models.append(camera["projection_model_parameters"])

        camera_model = VKCameraPanorama(input_cameras=input_cameras,
                                        stitch_params=data["stitching_parameters"],
                                        panorama_projection_models=panorama_projection_models,
                                        surface_name=_surface_model_name)

    # Load additional parameters if available
    if camera_model.surface_model is not None:
        if "homography" in data:
            camera_model.surface_model.homography = np.asarray(data["homography"])
            camera_model.surface_model.compute_inverse_homography()

        if "image_points" in data:
            camera_model.surface_model.image_points = np.asarray(data["image_points"])

        if "model_points" in data:
            camera_model.surface_model.model_points = np.asarray(data["model_points"])

    if "distortion_matrix" in data:
        camera_model.distortion_matrix = np.asarray(data["distortion_matrix"])

    if "camera_matrix" in data:
        camera_model.camera_matrix = np.asarray(data["camera_matrix"])

    if "focal_length" in data:
        camera_model.focal_length = data["focal_length"]

    return camera_model


def load_camera_model_from_json(path):
    """Parse json file for VKCamera object.

    Args:
        path (str): location of the json file.

    Returns:
        camera_model (VKCamera): model representation camera described by the data object.
    """
    assert os.path.exists(path), "JSON (camera) path does not exist: {0}".format(path)

    with open(path) as data_file:
        _data = json.load(data_file)
        return parse_camera_model_with_dict(_data)
