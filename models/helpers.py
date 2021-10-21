import cv2
import os

# SPORT CONSTANTS
VK_CAMERA_MODEL_UNKNOWN = -1
VK_CAMERA_MODEL_HOCKEY = 0
VK_CAMERA_MODEL_TENNIS = 1
VK_CAMERA_MODEL_NETBALL = 2
VK_CAMERA_MODEL_BASKETBALL = 3
VK_CAMERA_MODEL_SWIMMING = 4
VK_CAMERA_MODEL_SOCCER = 5
VK_CAMERA_MODEL_RUGBY = 6
VK_CAMERA_MODEL_NFL = 7


def sport_name_with_constant(sport):
    """Lookup table, returns sport name as a string for the sport id constant.

    Returns:
        (str): Sport name as string
    """
    if sport == VK_CAMERA_MODEL_UNKNOWN:
        return "Unknown"
    elif sport == VK_CAMERA_MODEL_HOCKEY:
        return "Hockey"
    elif sport == VK_CAMERA_MODEL_BASKETBALL:
        return "Basketball"
    elif sport == VK_CAMERA_MODEL_NETBALL:
        return "Netball"
    elif sport == VK_CAMERA_MODEL_RUGBY:
        return "Rugby"
    elif sport == VK_CAMERA_MODEL_SOCCER:
        return "Soccer"
    elif sport == VK_CAMERA_MODEL_SWIMMING:
        return "Swimming"
    elif sport == VK_CAMERA_MODEL_TENNIS:
        return "Tennis"
    elif sport == VK_CAMERA_MODEL_TENNIS:
        return "NFL"


def sport_constant_with_name(name):
    """Lookup table, returns sport constant as int for the sport name.

    Returns:
        (int): Sport id as constant
    """
    if name == "Hockey":
        return VK_CAMERA_MODEL_HOCKEY
    elif name == "Basketball":
        return VK_CAMERA_MODEL_BASKETBALL
    elif name == "Netball":
        return VK_CAMERA_MODEL_NETBALL
    elif name == "Rugby":
        return VK_CAMERA_MODEL_RUGBY
    elif name == "Soccer":
        return VK_CAMERA_MODEL_SOCCER
    elif name == "Swimming":
        return VK_CAMERA_MODEL_SWIMMING
    elif name == "Tennis":
        return VK_CAMERA_MODEL_TENNIS
    elif name == "NFL":
        return VK_CAMERA_MODEL_NFL
    else:
        return VK_CAMERA_MODEL_UNKNOWN


def surface_image_with_sport(sport):
    """Surface model image for current class model.

    Returns:
        (array): Returns a numpy array in RGB channel order
    """
    _sport = sport
    if _sport.__class__.__name__ == "int":
        _sport = sport_name_with_constant(sport)

    _path = "./models/surfaces/{:s}.png".format(_sport)
    assert os.path.exists(_path), "WTF!!  The surface image does not exist: {0}".format(_path)
    img = cv2.imread(_path)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img


def surface_properties_for_sport(sport):
    """Surface model properties for current class model.

    Returns:
        (dict): Returns a dictionary with model dimensions, offsets and real world scale
            for meters-to-pixels conversion,
            Width, Height and offsets are in meters.
            i.e. width*scale = the width of the surface image in pixels.
    """
    _sport = sport
    if _sport.__class__.__name__ == "str":
        _sport = sport_constant_with_name(sport)
    assert _sport >= 0, "WTF!! Valid sport constant not assigned..."

    if _sport == VK_CAMERA_MODEL_SWIMMING:
        return {
            "model_width": 50,
            "model_height": 25,
            "model_offset_x": 1,
            "model_offset_y": 1,
            "model_scale": 10
        }

    elif _sport == VK_CAMERA_MODEL_TENNIS:
        return {
            "model_width": 30,
            "model_height": 15,
            "model_offset_x": 1,
            "model_offset_y": 1,
            "model_scale": 50
        }

    elif _sport == VK_CAMERA_MODEL_HOCKEY:
        return {
            "model_width": 91,
            "model_height": 55,
            "model_offset_x": 5,
            "model_offset_y": 5,
            "model_scale": 10
        }

    elif _sport == VK_CAMERA_MODEL_NETBALL:
        return {
            "model_width": 31,
            "model_height": 15,
            "model_offset_x": 3,
            "model_offset_y": 3,
            "model_scale": 100
        }

    elif _sport == VK_CAMERA_MODEL_BASKETBALL:
        return {
            "model_width": 29.0,
            "model_height": 15.0,
            "model_offset_x": 1.8,
            "model_offset_y": 1.8,
            "model_scale": 40
        }

    # TODO - populate remaining sports..
    # elif _sport == VK_CAMERA_MODEL_RUGBY:
    #     pass

    else:
        return {"model_width": 50,
                "model_height": 25,
                "model_offset_x": 1,
                "model_offset_y": 1,
                "model_scale": 10}
