from cameras.camera_base import VKCamera
from cameras.camera_generic import VKCameraGenericDevice
from cameras.camera_file import VKCameraVideoFile
from cameras.camera_panoramic import VKCameraPanorama
from cameras.helpers.panorama import *
from cameras.helpers.camera_parser import load_camera_model_from_json as load_camera_model

print("################################################")
print("#\n#\tVK-CameraKit\n#")

try:
    from cameras.camera_blackmagic import VKCameraBlackMagicRAW
    print("#\t\t* Blackmagic BRAW format supported.")
except ImportError:
    print("#\t\t* The Blackmagic BRAW SDK is not supported.")

try:
    from cameras.camera_vimba import VKCameraVimbaDevice
    import vimba
    print("#\t\t* Vimba SDK Supported:", vimba.__version__)
except ImportError:
    print("#\t\t* Vimba library is not supported.")

print("#\n################################################\n")
