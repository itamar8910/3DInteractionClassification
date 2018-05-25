import os
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import tensorflow as tf
import cv2
sess = tf.Session() #Launch the graph in a session.

my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
DEEPGAZE_PATH = '/home/itamar/University/3dimagery/interactionClassification/deepgaze/'
# Load the weights from the configuration folders
# my_head_pose_estimator.load_roll_variables(os.path.realpath(os.path.join(DEEPGAZE_PATH, "etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf")))
my_head_pose_estimator.load_pitch_variables(os.path.realpath(os.path.join(DEEPGAZE_PATH, "etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf")))
my_head_pose_estimator.load_yaw_variables(os.path.realpath(os.path.join(DEEPGAZE_PATH, "etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf")))

def get_yaw_pitch(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (285, 285))
    pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
    yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
    print(yaw, pitch)
    return yaw, pitch
if __name__ == "__main__":
    print(get_yaw_pitch('/home/itamar/forks/gazr/pics/yaw_right.jpg'))
    print(get_yaw_pitch('/home/itamar/forks/gazr/pics/yaw_left.jpg'))
    print(get_yaw_pitch('/home/itamar/forks/gazr/pics/high_pitch.jpg'))
    print(get_yaw_pitch('/home/itamar/forks/gazr/pics/low_pitch.jpg'))
    print(get_yaw_pitch('/home/itamar/forks/gazr/pics/straight.jpg'))
    print(get_yaw_pitch('/home/itamar/forks/gazr/pics/1.jpg'))