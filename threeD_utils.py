import numpy as np


def line_from_center_triag(p1, p2, p3):
    """
    returns two points that define the line the start from the center of the triangle,
    in the direction of the normal to its plane
    :param p1:
    :param p2:
    :param p3:
    :return:
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    v1 = p3 - p1
    v2 = p2 - p1

    direction = np.cross(v1, v2)
    center = (p1 + p2 + p3) / 3.0

    t = 1
    return center, center + direction * t

def distance_line_point(line_p1, line_p2, point):
    return np.linalg.norm(np.cross(point - line_p1, point - line_p2)) / np.linalg.norm(line_p2 - line_p1)

def look_distance(lEye1, rEye1, nose1, lEye2, rEye2, nose2):
    """
    returns distance1->2, distance 2->1, between the looking direction of one person to the face of the other
    """
    return distance_line_point(*line_from_center_triag(lEye1, rEye1, nose1),
                               (np.array(lEye2) + np.array(rEye2) + np.array(nose2))/3.0), \
           distance_line_point(*line_from_center_triag(lEye2, rEye2, nose2),
                               (np.array(lEye1) + np.array(rEye1) + np.array(nose1))/3.0)


if __name__ == "__main__":
    lEye = (50, 50, 0)
    REye = (40, 50, 0)
    Nose = (45, 40, 0)
    line = line_from_center_triag(lEye, REye, Nose)
    print(line)
    test_point = 45, 460, 200
    print(distance_line_point(*line, test_point))