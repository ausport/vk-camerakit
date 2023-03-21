import cv2
import os
import math
import numpy as np

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

    _path = "./world_models/surfaces/{:s}.png".format(_sport)
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
            "model_width": 91.4,
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

def metric_surface_lines_with_sport(sport):

    _markings = []
    if sport == VK_CAMERA_MODEL_HOCKEY:

        pts = []
        pts.append((0.000, 0.000))
        # // left perimeter
        pts.append((0.000, 15.620))
        pts.append((-0.300, 15.620))
        pts.append((-0.300, 15.695))
        pts.append((0.000, 15.695))
        pts.append((0.000, 20.620))
        pts.append((-0.300, 20.620))
        pts.append((-0.300, 20.695))
        pts.append((0.000, 20.695))
        pts.append((0.000, 25.595))
        pts.append((-0.150, 25.595))
        pts.append((-0.150, 25.670))
        pts.append((0.000, 25.670))
        pts.append((0.000, 29.330))
        pts.append((-0.150, 29.330))
        pts.append((-0.150, 29.405))
        pts.append((0.000, 29.405))
        pts.append((0.000, 34.305))
        pts.append((-0.300, 34.305))
        pts.append((-0.300, 34.380))
        pts.append((0.000, 34.380))
        pts.append((0.000, 39.305))
        pts.append((-0.300, 39.305))
        pts.append((-0.300, 39.380))
        pts.append((0.000, 39.380))

        # // top left corner
        pts.append((0.000, 55.000))

        # // top perimeter
        pts.append((4.925, 55.000))
        pts.append((4.925, 55.300))
        pts.append((5.000, 55.300))
        pts.append((5.000, 55.000))
        pts.append((14.555, 55.000))
        pts.append((14.555, 55.300))
        pts.append((14.630, 55.300))
        pts.append((14.630, 55.000))
        pts.append((76.770, 55.000))
        pts.append((76.770, 55.300))
        pts.append((76.845, 55.300))
        pts.append((76.845, 55.000))
        pts.append((86.400, 55.000))
        pts.append((86.400, 55.300))
        pts.append((86.475, 55.300))
        pts.append((86.475, 55.000))

        # // top right corner
        pts.append((91.400, 55.000))

        # // right perimeter
        pts.append((91.400, 39.380))
        pts.append((91.700, 39.380))
        pts.append((91.700, 39.305))
        pts.append((91.400, 39.305))
        pts.append((91.400, 34.380))
        pts.append((91.700, 34.380))
        pts.append((91.700, 34.305))
        pts.append((91.400, 34.305))
        pts.append((91.400, 29.405))
        pts.append((91.550, 29.405))
        pts.append((91.550, 29.330))
        pts.append((91.400, 29.330))
        pts.append((91.400, 25.670))
        pts.append((91.550, 25.670))
        pts.append((91.550, 25.595))
        pts.append((91.400, 25.595))
        pts.append((91.400, 20.695))
        pts.append((91.700, 20.695))
        pts.append((91.700, 20.620))
        pts.append((91.400, 20.620))
        pts.append((91.400, 15.695))
        pts.append((91.700, 15.695))
        pts.append((91.700, 15.620))
        pts.append((91.400, 15.620))

        # // bottom right corner
        pts.append((91.400, 0.000))

        # // bottom perimeter
        pts.append((86.475, 0.000))
        pts.append((86.475, -0.300))
        pts.append((86.400, -0.300))
        pts.append((86.400, 0.000))
        pts.append((76.845, 0.000))
        pts.append((76.845, -0.300))
        pts.append((76.770, -0.300))
        pts.append((76.770, 0.000))
        pts.append((14.630, 0.000))
        pts.append((14.630, -0.300))
        pts.append((14.555, -0.300))
        pts.append((14.555, 0.000))
        pts.append((5.000, 0.000))
        pts.append((5.000, -0.300))
        pts.append((4.925, -0.300))
        pts.append((4.925, 0.000))
        _markings.append(pts)

        # left mid field
        pts = []
        pts.append((22.900, 0.075))
        pts.append((22.900, 54.925))
        pts.append((45.6625, 54.925))
        pts.append((45.6625, 0.075))
        _markings.append(pts)

        # right mid field
        pts = []
        pts.append((68.500, 0.075))
        pts.append((68.500, 54.925))
        pts.append((45.7375, 54.925))
        pts.append((45.7375, 0.075))
        _markings.append(pts)

        # // left inner circle
        innerRadius = 14.63 - 0.075

        pts = []
        pts.append((0.075, 25.670 - math.sqrt(innerRadius * innerRadius - 0.075 * 0.075)))

        dTheta = 0.02
        for theta in np.arange(math.atan(0.075 / innerRadius), math.pi / 2., dTheta):
            pts.append((0.000 + math.sin(theta) * innerRadius, 25.670 - math.cos(theta) * innerRadius))

        pts.append((innerRadius, 25.670))
        pts.append((innerRadius, 29.330))

        for theta in np.arange( math.pi / 2., math.atan(0.075 / innerRadius), -dTheta):
            pts.append((0.000 + math.sin(theta) * innerRadius, 29.330 + math.cos(theta) * innerRadius))

        pts.append((0.075, 29.330 + math.sqrt(innerRadius * innerRadius - 0.075 * 0.075)))
        _markings.append(pts)

        # // right inner circle
        pts = []
        pts.append((91.325, 25.670 - math.sqrt(innerRadius * innerRadius - 0.075 * 0.075)))

        for theta in np.arange(math.atan(0.075 / innerRadius), math.pi / 2., dTheta):
            pts.append((91.4 - math.sin(theta) * innerRadius, 25.670 - math.cos(theta) * innerRadius))

        pts.append((91.4 - innerRadius, 25.670))
        pts.append((91.4 - innerRadius, 29.330))

        for theta in np.arange( math.pi / 2., math.atan(0.075 / innerRadius), -dTheta):
            pts.append((91.4 - math.sin(theta) * innerRadius, 29.330 + math.cos(theta) * innerRadius))

        pts.append((91.325, 29.330 + math.sqrt(innerRadius * innerRadius - 0.075 * 0.075)))
        _markings.append(pts)

        # // left outer circle
        outerRadius = 14.63
        pts = []
        pts.append((0.075, 0.075))
        pts.append((0.075, 25.670 - math.sqrt(outerRadius * outerRadius - 0.075 * 0.075)))


        for theta in np.arange(math.atan(0.075 / outerRadius), math.pi / 2., dTheta):
            pts.append((0.000 + math.sin(theta) * outerRadius, 25.670 - math.cos(theta) * outerRadius))

        pts.append((outerRadius, 25.670))
        pts.append((outerRadius, 29.330))


        for theta in np.arange(math.pi / 2., math.atan(0.075 / outerRadius), -dTheta):
            pts.append((0.000 + math.sin(theta) * outerRadius, 29.330 + math.cos(theta) * outerRadius))

        pts.append((0.075, 29.330 + math.sqrt(outerRadius * outerRadius - 0.075 * 0.075)))
        pts.append((0.075, 54.925))
        pts.append((22.825, 54.925))
        pts.append((22.825, 0.075))
        _markings.append(pts)


        # // right outer circle
        pts = []
        pts.append((91.325, 0.075))
        pts.append((91.325, 25.670 - math.sqrt(outerRadius * outerRadius - 0.075 * 0.075)))

        for theta in np.arange(math.atan(0.075 / outerRadius), math.pi / 2., dTheta):
            pts.append((91.4 - math.sin(theta) * outerRadius, 25.670 - math.cos(theta) * outerRadius))

        pts.append((91.4 - outerRadius, 25.670))
        pts.append((91.4 - outerRadius, 29.330))

        for theta in np.arange( math.pi / 2., math.atan(0.075 / outerRadius), -dTheta):
            pts.append((91.4 - math.sin(theta) * outerRadius, 29.330 + math.cos(theta) * outerRadius))

        pts.append((91.325, 29.330 + math.sqrt(outerRadius * outerRadius - 0.075 * 0.075)))
        pts.append((91.325, 54.925))
        pts.append((68.575, 54.925))
        pts.append((68.575, 0.075))
        _markings.append(pts)

        penaltySpotRadius = 0.15 / 2.

        # // left penalty spot
        pts = []
        for i in range(0, 90):
            theta = i / 90.0 * math.pi * 2.0
            pts.append((math.cos(theta) * penaltySpotRadius + 6.475, math.sin(theta) * penaltySpotRadius + 27.5))
        _markings.append(pts)

        # // right penalty spot
        pts = []
        for i in range(0, 90):
            theta = i / 90.0 * math.pi * 2.0
            pts.append((91.4 - 6.475 + math.cos(theta) * penaltySpotRadius, math.sin(theta) * penaltySpotRadius + 27.5))
        _markings.append(pts)

        # // Dashed Circle Lines
        pts = []
        pts.append((71.770, 27.350))
        pts.append((71.770, 27.650))
        pts.append((71.695, 27.650))
        pts.append((71.695, 27.350))
        _markings.append(pts)

        pts = []
        pts.append((19.630, 27.350))
        pts.append((19.630, 27.650))
        pts.append((19.555, 27.650))
        pts.append((19.555, 27.350))
        _markings.append(pts)

        x0 = [0.000, 0.000, 91.400, 91.400]
        y0 = [29.330, 25.670, 29.330, 25.670]
        dx = [+1.0, +1.0, -1.0, -1.0]
        dy = [+1.0, -1.0, +1.0, -1.0]
        outer_dashed_radius = 14.63 + 5.00
        inner_dashed_radius = outer_dashed_radius - 0.075

        for i in range(0, 9):
            starting_arc_length = i * 3.3 + 1.32
            stopping_arc_length = starting_arc_length + 0.3

            # // compute the starting and stopping angles.
            starting_theta = starting_arc_length / outer_dashed_radius
            stopping_theta = stopping_arc_length / outer_dashed_radius

            starting_sin = math.sin(starting_theta)
            starting_cos = math.cos(starting_theta)
            stopping_sin = math.sin(stopping_theta)
            stopping_cos = math.cos(stopping_theta)

            for quadrant in range(0, 4):
                pts = []
                pts.append((x0[quadrant] + dx[quadrant] * outer_dashed_radius * starting_cos, y0[quadrant] + dy[quadrant] * outer_dashed_radius * starting_sin ))
                pts.append((x0[quadrant] + dx[quadrant] * outer_dashed_radius * stopping_cos, y0[quadrant] + dy[quadrant] * outer_dashed_radius * stopping_sin ))
                pts.append((x0[quadrant] + dx[quadrant] * inner_dashed_radius * stopping_cos, y0[quadrant] + dy[quadrant] * inner_dashed_radius * stopping_sin ))
                pts.append((x0[quadrant] + dx[quadrant] * inner_dashed_radius * starting_cos, y0[quadrant] + dy[quadrant] * inner_dashed_radius * starting_sin ))
                _markings.append(pts)

    elif sport == VK_CAMERA_MODEL_TENNIS:

        # // outer perimeter
        pts = []
        pts.append((0.000, 0.000))
        pts.append((0.000, 10.973))
        pts.append((23.774, 10.973))
        pts.append((23.774, 0.0003))
        _markings.append(pts)

        # // top doubles interior
        pts = []
        pts.append((0.050, 10.923))
        pts.append((23.724, 10.923))
        pts.append((23.724,  9.601))
        pts.append((0.050,  9.601))
        _markings.append(pts)

        # // bottom doubles interior
        pts = []
        pts.append((0.050,  1.372))
        pts.append((23.724,  1.372))
        pts.append((23.724,  0.050))
        pts.append((0.050,  0.050))
        _markings.append(pts)

        # // top service box interior
        pts = []
        pts.append((5.536,  9.551))
        pts.append((18.238,  9.551))
        pts.append((18.238,  5.5115))
        pts.append((5.536,  5.5115))
        _markings.append(pts)

        # // bottom service box interior
        pts = []
        pts.append((5.536,  5.4615))
        pts.append((18.238,  5.461))
        pts.append((18.238,  1.422))
        pts.append((5.536,  1.422))
        _markings.append(pts)

        # // left base court interior
        pts = []
        pts.append((0.050,  9.551))
        pts.append((5.486,  9.551))
        pts.append((5.486,  1.422))
        pts.append((0.050,  1.422))
        pts.append((0.050,  5.4615))
        pts.append((0.150,  5.4615))
        pts.append((0.150,  5.511))
        pts.append((0.050,  5.5115))
        _markings.append(pts)

        # // right base court interior
        pts = []
        pts.append((18.288,  9.551))
        pts.append((23.724,  9.551))
        pts.append((23.724,  5.511))
        pts.append((23.624,  5.5115))
        pts.append((23.624,  5.4615))
        pts.append((23.724,  5.4615))
        pts.append((23.724,  1.422))
        pts.append((18.288,  1.422))
        _markings.append(pts)

    return _markings
