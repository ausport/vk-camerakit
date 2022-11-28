"""General utilities for camera geometry"""


def ray_intersection(line1, line2):
    """Estimates a point where two lines should intersect.

    Args:
        line1 (list): a list of point coordinates for a line (x1, y1, x2, y2).
        line2 (list): a list of point coordinates for a line (x1, y1, x2, y2).

    Returns:
        (x,y): intersection point.
    """

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('rays do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
