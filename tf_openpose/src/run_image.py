import argparse
import logging
import time

import cv2
import numpy as np
from random import randint
from tf_openpose.src.estimator import TfPoseEstimator
from tf_openpose.src.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--img', type=str, default='')
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    # cam = cv2.VideoCapture(args.camera)
    # ret_val, image = cam.read()
    # logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    def show_for_image(img):
        image = cv2.imread(img)

        logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        logger.debug('image process+')
        humans = e.inference(image)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        toshow = cv2.resize(image, (640, 480))
        name = 'tf-pose-estimation result ' + str(randint(0,100))
        cv2.imshow(name, toshow)
        cv2.imwrite('../' + name + '.jpeg', image)
       
    show_for_image('/home/itamar/University/3dimagery/interactionClassification/tf_openpose/images2/notouch1.png')
    show_for_image('/home/itamar/University/3dimagery/interactionClassification/tf_openpose/images2/notouch2.png')
    # show_for_image('/home/itamar/University/3dimagery/interactionClassification/tf_openpose/images2/touch5.png')
    # show_for_image('tf_openpose/images/IMG_0339.jpg')
    # show_for_image('tf_openpose/images/IMG_0342.jpg')
    # show_for_image('tf_openpose/images/IMG_0341.jpg')
    # show_for_image('tf_openpose/images/IMG_0340.jpg')
    cv2.waitKey(0)
        # fps_time = time.time()
        # if cv2.waitKey(1) == 27:
        #     break
        # logger.debug('finished+')

    cv2.destroyAllWindows()
