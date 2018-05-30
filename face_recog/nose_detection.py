from collections import OrderedDict

import cv2
import imutils
import dlib
import numpy as np

predictor = dlib.shape_predictor('face_recog/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])

def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates

def pretty_print(landmarks):
    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = landmarks[j:k]
        print(name)
        print(pts)
        print(name)

def get_nose_from_face_img(face_img_path):
    image = cv2.imread(face_img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, dlib.rectangle(0, 0, image.shape[1], image.shape[0]))
    landmarks = shape_to_numpy_array(shape)
    point_center = np.mean(landmarks[27:35], axis = 0)
    return [point_center[0] / image.shape[1], point_center[1] / image.shape[0]]
    # print(shape)

def draw_nose_from_face_img(face_img_path):
    image = cv2.imread(face_img_path)
    nose_rel_x, nose_rel_y = get_nose_from_face_img(face_img_path)
    print(nose_rel_x, nose_rel_y)
    nose_x = int(nose_rel_x * image.shape[1])
    nose_y = int(nose_rel_y * image.shape[0])
    cv2.rectangle(image, (nose_x - 5, nose_y - 5), (nose_x + 5, nose_y + 5), color=(0,0,0), thickness=-1)
    cv2.imshow('img', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def nose_from_img(img_path):
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    print(rects)
    assert len(rects) == 1
    rect = rects[0]
    shape = predictor(gray, rect)
    landmarks = shape_to_numpy_array(shape)
    point_center = np.mean(landmarks[27:35], axis=0)
    print(point_center)
    # cv2.rectangle(image, rect[0], rect[1], (255, 255, 255), thickness=2)
    cv2.imshow('img', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    draw_nose_from_face_img('/home/itamar/Pictures/a.png')
    # draw_nose_from_face_img('/home/itamar/University/3dimagery/interactionClassification/tmp/face_exact.png')
    # print(draw_nose_from_face_img('dummy/Detect-Facial-Features/images/face4.png'))
    # print(nose_from_img('/home/itamar/University/3dimagery/interactionClassification/deepgaze/examples/ex_cnn_head_pose_axes/2.jpg'))