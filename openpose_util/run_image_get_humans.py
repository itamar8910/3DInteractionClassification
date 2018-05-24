import numpy as np

from tf_openpose.src.networks import get_graph_path, model_wh
from tf_openpose.src.estimator import TfPoseEstimator
import cv2
from PIL import Image, ImageDraw


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
    w, h = model_wh('432x368')

    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
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
    body_parts= ['REye', 'LEye', 'Nose', 'LShoulder', 'RShoulder', 'RHip', 'LHip']
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
            elif np.mean(bgr) > 115:
                return 'ALON'
            else:
                return 'ITAMAR'
        return classify(bgr)
    avg_colors = []
    if 'LShoulder' in human.keys() and human['LShoulder'] is not None:
        avg_colors.append(get_avg_color_around_point(cvimg, human['LShoulder']['x'], human['LShoulder']['y']))

    if 'RShoulder' in human.keys() and human['RShoulder'] is not None:
        avg_colors.append(get_avg_color_around_point(cvimg, human['RShoulder']['x'], human['RShoulder']['y']))

    if len(avg_colors) > 0:
        if bgr_to_identity(avg_colors[0]) == 'GIL':
            return 'GIL'
        if len(avg_colors) > 1 and bgr_to_identity(avg_colors[1]) == 'GIL':
            return 'GIL'
        return bgr_to_identity(np.mean(avg_colors, axis = 0))
    else:
        return bgr_to_identity(get_mean_human(human))



def get_humans_data(img_path):
    cvimg = cv2.imread(img_path)
    humans_keypoints = get_humans_keypoints(img_path)
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
        humans_data.append(human_data)
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

def draw_keypoints(img_path):
    # keypoints = get_humans_keypoints(img_path)
    huamans = get_humans_data(img_path)
    # print(keypoints)


    cvimg = cv2.imread(img_path)
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for human in huamans:
        # if 'LShoulder' in human.keys():
        #     avg_color = get_avg_color_around_point(cvimg, human['LShoulder']['x'], human['LShoulder']['y'])
        #     draw.text([(human['LShoulder']['x'], human['LShoulder']['y'] + 30)], str(avg_color), fill=(0,0,255))

        # human_dist = get_human_distance(human)
        # # print(get_mean_human(human))
        # print(human_dist)
        draw.text([tuple(human['center'])], str(human['distance']), fill=(0, 0, 255))
        draw.text([tuple([human['center'][0], human['center'][1]-20])], str(human['identity']), fill=(0, 255, 0))

        for part in human['keypoints'].keys():
            if human['keypoints'][part] is not None:
                x = int(human['keypoints'][part]['x'])
                y = int(human['keypoints'][part]['y'])
                draw.rectangle([(x, y), (x + 20, y + 20)])
                draw.text([(x, y)], part)
            else:
                print(part, ' is missing')
    img.show()

if __name__ == "__main__":
    img_path = 'calibration/frames/cam0/frame_40.000.jpg'
    draw_keypoints(img_path)
