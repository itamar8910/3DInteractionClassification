
from tf_openpose.src.networks import get_graph_path, model_wh
from tf_openpose.src.estimator import TfPoseEstimator
import cv2


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

def get_humans(img_path):
    w, h = model_wh('432x368')

    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
    image = cv2.imread(img_path)
    humans = e.inference(image)
    return humans

if __name__ == "__main__":
    img_path = 'tf_openpose/images/IMG_0339.jpg'
    print(get_humans(img_path))
