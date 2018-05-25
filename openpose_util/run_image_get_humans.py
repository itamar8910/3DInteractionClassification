import numpy as np
from os import path

from Tiny_Faces_in_Tensorflow.get_faces import get_faces
from tf_openpose.src.networks import get_graph_path, model_wh
from tf_openpose.src.estimator import TfPoseEstimator
import cv2
from PIL import Image, ImageDraw

from threeD_utils import line_from_center_triag, distance_line_point, get_face_dir

POSE_COCO_BODY_PARTS = [
     [0,  "Nose"],
     [1,  "Neck"],
     [2,  "RShoulder"],
     [3,  "RElbow"],
     [4,  "RWrist"],
     [5,  "LShoulder"],
     [6,  "LElbow"],
     [7,  "LWrist"],
     [8,  "RHip"],
     [9,  "RKnee"],
     [10, "RAnkle"],
     [11, "LHip"],
     [12, "LKnee"],
     [13, "LAnkle"],
     [14, "REye"],
     [15, "LEye"],
     [16, "REar"],
     [17, "LEar"],
     [18, "Background"],
]

body_part_to_index = {part : index for index , part in POSE_COCO_BODY_PARTS}
print(body_part_to_index)

def get_humans(image):
    """
    :param image: Image fater cv2.imread(image_path)
    :return:
    """
    MODEL_TO_SIZE = {
        'cmu': '656x368',
        'mobilenet_thin': '432x368'
    }
    MODEL = 'mobilenet_thin'
    w, h = model_wh(MODEL_TO_SIZE[MODEL])

    e = TfPoseEstimator(get_graph_path(MODEL), target_size=(w, h))
    humans = e.inference(image)
    return humans

def get_humans_keypoints(img_path):
    """
    returns list of left eye, right eye, nose coordinates (for each human). None if keypoint wasn't detected.
    :param img_path:
    :return:
    """
    image = cv2.imread(img_path)

    def body_part_to_x_y(body_part):
        return {'x': image.shape[1] * body_part.x, 'y': image.shape[0] * body_part.y}

    def index_to_body_part_x_y(human, body_part_index):
        return body_part_to_x_y(human.body_parts[body_part_index]) if body_part_index in human.body_parts.keys() else None

    humans = get_humans(image)
    body_parts= ['REye', 'LEye', 'Nose', 'LShoulder', 'RShoulder', 'RHip', 'LHip', 'LEar', 'REar', 'Neck']
    return [{part: index_to_body_part_x_y(human, body_part_to_index[part]) for part in body_parts} for human in humans]

def get_human_identity(cvimg, human):
    def bgr_to_identity(bgr):
        GIL_MEAN_BGR = [108.5, 60.8125, 24.875]
        ITAMAR_MEAN_BGR = [74.46666667, 60.8, 51.66666667]
        ALON_MEAN_BGR = [184.22727273, 192.59090909, 196.54545455]

        def classify(bgr):
            print(bgr)
            if bgr[0] / bgr[2] > 2.0:
                return 'GIL'
            elif np.mean(bgr) < 60:
                return 'ITAMAR'
            elif np.mean(bgr) > 100:
                return 'ALON'
            else:
                return 'ITAMAR'
        return classify(bgr)
    avg_colors = []
    if 'LShoulder' in human.keys() and human['LShoulder'] is not None:
        avg_colors.append(get_avg_color_around_point(cvimg, human['LShoulder']['x'], human['LShoulder']['y']+10))

    if 'RShoulder' in human.keys() and human['RShoulder'] is not None:
        avg_colors.append(get_avg_color_around_point(cvimg, human['RShoulder']['x'], human['RShoulder']['y']+10))

    if len(avg_colors) > 0:
        if bgr_to_identity(avg_colors[0]) == 'GIL':
            return 'GIL'
        if len(avg_colors) > 1 and bgr_to_identity(avg_colors[1]) == 'GIL':
            return 'GIL'
        if bgr_to_identity(avg_colors[0]) == 'ITAMAR':
            return 'ITAMAR'
        if len(avg_colors) > 1 and bgr_to_identity(avg_colors[1]) == 'ITAMAR':
            return 'ITAMAR'
        return bgr_to_identity(np.mean(avg_colors, axis = 0))
    else:
        return bgr_to_identity(get_mean_human(human))

def human_to_ears_nose_depth(human):

    DEPTH_FACTOR = 10
    depth = human['distance']

    if 'LEye' in human['keypoints'].keys() and human['keypoints']['LEye'] is not None\
            and 'REye' in human['keypoints'].keys() and human['keypoints']['REye'] is not None:
        dist_LEye_to_nose = np.linalg.norm([keypoint_to_xy(human['keypoints']['LEye']),
                                            keypoint_to_xy(human['keypoints']['Nose'])])
        dist_REye_to_nose = np.linalg.norm([keypoint_to_xy(human['keypoints']['REye']),
                                            keypoint_to_xy(human['keypoints']['Nose'])])

        human['LEye'] = {
            'x': human['keypoints']['LEye']['x'],
            'y': human['keypoints']['LEye']['y'],
            'z': depth}
        human['REye'] = {
            'x': human['keypoints']['REye']['x'],
            'y': human['keypoints']['REye']['y'],
            'z': depth}
        human['Nose'] = {
            'x': human['keypoints']['Nose']['x'],
            'y': human['keypoints']['Nose']['y'],
            'z': depth}

        if dist_LEye_to_nose > dist_REye_to_nose:
            ratio = dist_LEye_to_nose / dist_REye_to_nose
            REye_depth = depth + ratio * DEPTH_FACTOR
            human['REye']['z'] = REye_depth
        else:
            ratio = dist_REye_to_nose / dist_LEye_to_nose
            LEye_depth = depth + ratio * DEPTH_FACTOR
            human['LEye']['z'] = LEye_depth


def is_facing_forward(human_keypoints):
    return 'Nose' in human_keypoints.keys() and human_keypoints['Nose'] is not None


def keypoints_distance(kp1, kp2):
    return np.linalg.norm(np.array(keypoint_to_xy(kp1)) - np.array(keypoint_to_xy(kp2)))

def get_human_face_width(human_keypoints):
    ret_val = None

    if human_keypoints['LEar'] is not None and human_keypoints['REar'] is not None:
        ret_val = keypoints_distance(human_keypoints['LEar'], human_keypoints['REar'])

    elif human_keypoints['LEye'] is not None and human_keypoints['REye'] is not None:
        ret_val = keypoints_distance(human_keypoints['LEye'], human_keypoints['REye'])

    elif human_keypoints['LShoulder'] is not None and human_keypoints['RShoulder'] is not None:
        shoulder_face_width_ratio = 1.5
        ret_val = keypoints_distance(human_keypoints['LShoulder'], human_keypoints['RShoulder']) / shoulder_face_width_ratio
    else:
        print("WARNING: couldn't detect human face width, returning default")
        default_width = 150
        ret_val = default_width

    min_width = 50
    return max(ret_val, min_width)

def get_face_img(cvimg, face_bbox):
    # face_center_x, face_center_y = get_human_face_center_xy(human_keypoints)
    # face_width = get_human_face_width(human_keypoints)
    # face_width *= 1
    padding_factor = 2
    face_width = face_bbox[2] - face_bbox[0]
    face_height = face_bbox[3] - face_bbox[1]
    face_img = cvimg[max(0, int(face_bbox[1] - (face_height/2)*padding_factor )): min(int(face_bbox[3] + (face_height/2)*padding_factor), cvimg.shape[0]),
               max(0, int(face_bbox[0] - (face_width/2)*padding_factor )): min(int( face_bbox[2] + (face_width/2)*padding_factor), cvimg.shape[1])]
    return cv2.resize(face_img, (500, 500))
    


def get_face_direction(cvimg, face_bbox, keypoints):
    face_img = get_face_img(cvimg, face_bbox)
    tmp_path = 'tmp/face.png'
    cv2.imwrite(tmp_path, face_img)
    face_dir_gazr = get_face_dir(path.abspath(tmp_path))
    if face_dir_gazr is not None:
        return face_dir_gazr

    print("GAZR didn't recognize, falling back on openpose heuristic")
    # heuristic with openpose
    if keypoints['REye'] is not None and keypoints['LEye'] is not None:
        return (0, 0, 0)
    if keypoints['REye'] is not None:
        return (1, 0, 0)
    elif keypoints['LEye'] is not None:
        return (1, 0, 0)

def find_closest_face(faces, point):
    return min(faces, key = lambda face : np.linalg.norm([np.mean([face[0], face[2]]), np.mean([face[1], face[3]])] - np.array(point)) )


def get_humans_data(img_path):
    cvimg = cv2.imread(img_path)
    humans_keypoints = get_humans_keypoints(img_path)
    faces = get_faces(img_path)
    humans_data = []
    for human_keypoints in humans_keypoints:
        human_data = {}
        human_data['keypoints'] = human_keypoints
        human_distance = get_human_distance(human_keypoints)
        human_identity = get_human_identity(cvimg, human_keypoints)
        print('identity:', human_identity)
        human_data['distance'] = human_distance
        human_data['identity'] = human_identity
        human_data['center'] = get_mean_human(human_keypoints)
        if human_data['keypoints']['Nose'] is not None:
            human_data['face_bbox'] = find_closest_face(faces, keypoint_to_xy(human_data['keypoints']['Nose']))
            human_data['face_dir'] = get_face_direction(cvimg, human_data['face_bbox'], human_data['keypoints'])
        humans_data.append(human_data)
        # if is_facing_forward(human_keypoints):
        #
        #     face_dir = get_face_direction(cvimg, human_data['keypoints'])
        #     if face_dir is not None:
        #         human_data['face_dir'] = face_dir
    return humans_data

def get_avg_color_around_point(img, x, y, width=5):
    print('color around:',x,y, np.mean(img[int(y-width) : int(y+width), int(x-width):int(x+width)], axis=(0,1)))

    return np.mean(img[int(y-width) : int(y+width), int(x-width):int(x+width)], axis=(0,1))

def keypoint_to_xy(keypoint):
    return keypoint['x'], keypoint['y']

def get_human_shoulder_hip_distance(human):
    shoulders = ['LShoulder', 'RShoulder']
    hips = ['LHip', 'RHip']
    mean_shoulder = np.mean([keypoint_to_xy(human[x]) for x in shoulders if human[x] is not None], axis = 0)
    mean_heap = np.mean([keypoint_to_xy(human[x]) for x in hips if human[x] is not None], axis = 0)
    print('shoulder:',  mean_shoulder)
    print('heap:', mean_heap)
    return np.linalg.norm(mean_heap - mean_shoulder)

def get_human_distance(human, MUL_FACT = 10e3):
    return (1 / get_human_shoulder_hip_distance(human)) * MUL_FACT

def get_mean_human(human):
    return np.mean([keypoint_to_xy(kp) for kp in human.values() if kp is not None], axis = 0)

def get_human_face_center_xy(human):
    assert 'keypoints' in human.keys()

    human_keypoints = human['keypoints']
    if 'face_bbox' in human.keys():
        face_bbox = human['face_bbox']
        return ((face_bbox[0] + face_bbox[2]) / 2.0 , (face_bbox[1] + face_bbox[3]) / 2.0)

    if 'LEar' in human_keypoints.keys() and human_keypoints['LEar'] is not None \
            and 'REar' in human_keypoints.keys() and human_keypoints['REar'] is not None:
        return ((human_keypoints['LEar']['x'] + human_keypoints['REar']['x'])/2.0,
                     (human_keypoints['LEar']['y'] + human_keypoints['REar']['y']) / 2.0,
                )
    elif 'Nose' in human_keypoints.keys() and human_keypoints['Nose'] is not None:
        return (human_keypoints['Nose']['x'], human_keypoints['Nose']['y'])

    elif human_keypoints['Neck'] is not None:
        hips = ['LHip', 'RHip']
        if any([human_keypoints[x] is not None for x in hips]):
            mean_heap = np.mean([keypoint_to_xy(human_keypoints[x]) for x in hips if human_keypoints[x] is not None], axis=0)
            neck_hip_distance = np.linalg.norm(keypoint_to_xy(human_keypoints['Neck']) - mean_heap)
            neck_face_dist_ratio = 0.25
            return (human_keypoints['Neck']['x'], human_keypoints['Neck']['y'] - neck_hip_distance*neck_face_dist_ratio)
        else:
            neck_face_offset = 50
            return (human_keypoints['Neck']['x'], human_keypoints['Neck']['y'] - neck_face_offset)

    elif human_keypoints['LShoulder'] is not None and human_keypoints['RShoulder'] is not None:
        shoulder_face_offset = 100
        return ((human_keypoints['LShoulder']['x'] +human_keypoints['RShoulder']['x'])/2.0,
                     (human_keypoints['LShoulder']['y'] + human_keypoints['RShoulder']['y']) / 2.0 - shoulder_face_offset,
                )

    else:
        raise Exception("couldn't get human face center")

def get_human_face_center_xyz(human):
    x_center, y_center = get_human_face_center_xy(human)
    return (x_center, y_center, human['distance'])


def get_looking_at_on_another(human1, human2):
    assert is_facing_forward(human1['keypoints']) and 'face_dir' in human1.keys()

    human1_face_center = get_human_face_center_xyz(human1)
    huamn1_face_line = (np.array(human1_face_center, dtype=np.float32), np.array(human1_face_center, dtype=np.float32)
                        + 10 * np.array(human1['face_dir'], dtype=np.float32))

    human2_center = get_human_face_center_xyz(human2)

    look_distance = distance_line_point(*huamn1_face_line, human2_center)
    print('look distnace:', look_distance)


def draw_keypoints(img_path):
    # keypoints = get_humans_keypoints(img_path)
    humans = get_humans_data(img_path)
    # for human in huamans:
    #     human_to_ears_nose_depth(human)
    #     print(human)
    # print(huamans)

    gil = [human for human in humans if human['identity'] == 'GIL'][0]
    itamar = [human for human in humans if human['identity'] == 'ITAMAR'][0]
    print('gil looking at itamar:')
    get_looking_at_on_another(gil, itamar)
    print('itamar looking at gil:')
    get_looking_at_on_another(itamar, gil)
    #
    #
    # # print(keypoints)


    cvimg = cv2.imread(img_path)
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for human in humans:
        print(human['identity'])
        print(human['keypoints'])
        # if 'LShoulder' in human.keys():
        #     avg_color = get_avg_color_around_point(cvimg, human['LShoulder']['x'], human['LShoulder']['y'])
        #     draw.text([(human['LShoulder']['x'], human['LShoulder']['y'] + 30)], str(avg_color), fill=(0,0,255))

        # human_dist = get_human_distance(human)
        # # print(get_mean_human(human))
        # print(human_dist)
        draw.text([tuple(human['center'])], str(human['distance']), fill=(0, 0, 255))
        draw.text([tuple([human['center'][0], human['center'][1]-40])], str(human['identity']), fill=(0, 255, 0))

        face_center = get_human_face_center_xy(human)
        draw.rectangle([(face_center[0] - 5, face_center[1] - 5), (face_center[0] + 5, face_center[1] + 5)], fill=(255, 0, 0))

        if 'face_bbox' in human.keys():
            draw.rectangle(((human['face_bbox'][0], human['face_bbox'][1]), (human['face_bbox'][2],human['face_bbox'][3])))
        if 'face_dir' in human.keys() and human['face_dir'] is not None:
            face_center_x, face_center_y = get_human_face_center_xy(human)
            # draw.rectangle([(face_center_x - 20, face_center_y - 20), (face_center_x + 20, face_center_y + 20)])
            look_dir = human['face_dir']
            look_ahead_x = face_center_x + look_dir[0] * 100
            look_ahead_y = face_center_y + look_dir[1] * 100
            draw.line([(face_center_x, face_center_y), (look_ahead_x, look_ahead_y)])
        #
        for part in human['keypoints'].keys():
            if human['keypoints'][part] is not None:
                x = int(human['keypoints'][part]['x'])
                y = int(human['keypoints'][part]['y'])
                draw.rectangle([(x-2, y-2), (x + 2, y + 2)], fill=(0,255,0))
                # draw.text([(x, y)], part)
            else:
                print(part, ' is missing')
    img.show()

if __name__ == "__main__":
    # img_path = 'calibration/frames/cam0/frame_164.700.jpg'
    img_path = 'calibration/frames/cam0/frame_160.400.jpg'
    # print(get_faces(img_path))
    # img = cv2.imread(img_path)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    draw_keypoints(img_path)
