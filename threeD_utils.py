import json

import numpy as np
import math

from subprocess import Popen, PIPE


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

def get_dir_vector_from_yaw_pitch(yaw, pitch):
    yaw -= 180
    pitch -= 180

    xdir = - math.sin(yaw * math.pi / 180.0)
    ydir = math.sin(pitch * math.pi / 180.0)
    zdir = - math.cos(pitch * math.pi / 180.0) * math.cos(yaw * math.pi / 180.0)

    return (xdir, ydir, zdir)

def draw_gaze_dir(img_path, yaw, pitch):
    dir = get_dir_vector_from_yaw_pitch(yaw, pitch)
    import cv2
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    start_x = width / 2
    start_y = height/2
    linelen = 1000
    end_x = start_x - linelen * dir[0]
    end_y = start_y + linelen * dir[1]
    cv2.line(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color=(255,0,0), thickness=5)
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_face_dir(full_img_path):
    cmd = "/home/itamar/forks/gazr/build/gazr_show_head_pose --model /home/itamar/forks/gazr/share/shape_predictor_68_face_landmarks.dat --image {}".format(full_img_path)
    out = Popen(cmd, stdout=PIPE, shell=True).communicate()[0].decode()
    out_dict = eval(out.strip())
    print(out_dict)
    if len(out_dict) != 1:
        return None
    else:
        return get_dir_vector_from_yaw_pitch(out_dict['face_0']['yaw'], out_dict['face_0']['pitch'])


if __name__ == "__main__":
    print(get_face_dir('/home/itamar/University/3dimagery/interactionClassification/deepgaze/examples/ex_cnn_head_pose_axes/2.jpg'))
    exit()
    print('yaw left:', get_face_dir('/home/itamar/forks/gazr/pics/yaw_left.jpg'))
    print('yaw right:', get_face_dir('/home/itamar/forks/gazr/pics/yaw_right.jpg'))
    # exit()
    # draw_gaze_dir('/home/itamar/forks/gazr/pics/yaw_left.jpg', 154.2, 180.1)
    # exit()
    # print(get_dir_vector_from_yaw_pitch(200, 181.3))
    # print(get_dir_vector_from_yaw_pitch(154.2, 180.1))
    # print(get_dir_vector_from_yaw_pitch(173.4, 197.0))
    # print(get_dir_vector_from_yaw_pitch(175.1, 183.2))
    # lEye = (50, 50, 0)
    # REye = (40, 50, 0)
    # Nose = (45, 40, 0)
    # line = line_from_center_triag(lEye, REye, Nose)
    # print(line)
    # test_point = 45, 460, 200
    # print(distance_line_point(*line, test_point))