
from tf_openpose.src.networks import get_graph_path, model_wh
from tf_openpose.src.estimator import TfPoseEstimator
import cv2
from PIL import Image, ImageDraw

"""
// POSE_COCO_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "RHip"},
//     {9,  "RKnee"},
//     {10, "RAnkle"},
//     {11, "LHip"},
//     {12, "LKnee"},
//     {13, "LAnkle"},
//     {14, "REye"},
//     {15, "LEye"},
//     {16, "REar"},
//     {17, "LEar"},
//     {18, "Background"},
// }
"""

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
    return [{'LEye': index_to_body_part_x_y(human, 15),
             'REye': index_to_body_part_x_y(human, 14),
             'Nose': index_to_body_part_x_y(human, 0)} for human in humans]

def draw_keypoints(img_path):
    keypoints = get_humans_keypoints(img_path)
    print(keypoints)


    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for human in keypoints:
        for part in human.keys():
            if human[part] is not None:
                x = int(human[part]['x'])
                y = int(human[part]['y'])
                draw.rectangle([(x, y), (x + 20, y + 20)])
                draw.text([(x, y)], part)
            else:
                print(part, ' is missing')
    img.show()

if __name__ == "__main__":
    img_path = 'dummy/test4.png'
    draw_keypoints(img_path)
