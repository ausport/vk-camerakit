import vmbpy

import cameras.helpers.camera_parser
from cameras.camera_base import VKCamera
from cameras.camera_generic import VKCameraGenericDevice
from cameras.camera_file import VKCameraVideoFile
from cameras.camera_panoramic import VKCameraPanorama
from cameras.camera_vimba import VKCameraVimbaDevice
from cameras.helpers.panorama import *
from cameras.helpers.camera_parser import load_camera_model_from_json as load_camera_model
from cameras.camera_vimba import VIMBA_CAPTURE_MODE_ASYNCRONOUS, VIMBA_CAPTURE_MODE_SYNCRONOUS, VIMBA_INSTANCE

VK_ROTATE_NONE = -1
VK_ROTATE_90_CLOCKWISE = 0
VK_ROTATE_180 = 1
VK_ROTATE_90_COUNTERCLOCKWISE = 2

print("################################################")
print("#\n#\tVK-CameraKit\n#")

try:
    from cameras.camera_blackmagic import VKCameraBlackMagicRAW
    print("#\t\t* Blackmagic BRAW format supported.")
except ImportError:
    print("#\t\t* The Blackmagic BRAW SDK is not supported.")

try:
    from cameras.camera_vimba import VKCameraVimbaDevice
    from cameras.helpers.vimba_utilities import enumerate_vimba_devices, get_camera
    import vmbpy as vimba
    print("#\t\t* Vimba SDK Supported:", vimba.__version__)
except ImportError:
    print("#\t\t* Vimba library is not supported.")

print("#\n################################################\n")


def camera_with_calibration_file(calibration_file):
    return cameras.helpers.camera_parser.load_camera_model_from_json(path=calibration_file)
