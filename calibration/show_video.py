import cv2
import numpy as np

captures = [
    cv2.VideoCapture('footage/footage_synched/footage1.mp4'),
    cv2.VideoCapture('footage/footage_synched/footage2.mp4'),
    cv2.VideoCapture('footage/footage_synched/footage3.mp4'),
    cv2.VideoCapture('footage/footage_synched/footage4.mp4'),

]

GIL_MEAN_BGR = [108.5, 60.8125, 24.875]
ITAMAR_MEAN_BGR = [74.46666667, 60.8, 51.66666667]
ALON_MEAN_BGR = [184.22727273, 192.59090909, 196.54545455]

def classify(bgr):
    if bgr[0] / bgr[2] > 2.0:
        return 'GIL'
    elif np.mean(bgr) > 115:
        return 'ALON'
    else:
        return 'ITAMAR'
# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.

fpss = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
print(fpss)

PAUSE = False
DRAW_TIME = True
SAVE_FRAMES = True
start_save_sec, end_save_sec = 150, 165

frame_i = 0
cam_imgs = [None for i in captures]
clicks_rgb = []
def get_click_rgb(event, x, y, flags, args):
    # print('mouse event')
    if event == cv2.EVENT_LBUTTONDOWN:
        cam_i = args[0]
        clicks_rgb.append(cam_imgs[cam_i][y, x])
        print('mouse click:{},{}'.format(x, y))
        print('rgb {}: {}'.format(cam_i, cam_imgs[cam_i][y, x]))
        print(np.mean(clicks_rgb, axis = 0))
        print(classify(cam_imgs[cam_i][y, x]))


for i in range(len(captures)):
    cv2.namedWindow('some{}'.format(i))
    cv2.setMouseCallback('some{}'.format(i), get_click_rgb, param=(i, ))

while True:
    for cap_i, cap in enumerate(captures):
        frame_sec = frame_i / fpss[cap_i]

        ret, img = cap.read()

        img  = cv2.resize(img, (600, 400))
        cam_imgs[cap_i] = img
        # cv2.circle(img, (50, 200), 20, (0, 0, 0), thickness=-1)

        if SAVE_FRAMES and start_save_sec <= frame_sec <= end_save_sec:
            cv2.imwrite('calibration/frames/cam{}/frame_{:.3f}.jpg'.format(cap_i, frame_sec), img)
        if DRAW_TIME:
            cv2.putText(img, "time:{:.2f}".format(frame_sec), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow('some{}'.format(cap_i), img)
        # cv2.resizeWindow('some{}'.format(cap_i), 100, 100)
    frame_i += 1

    if PAUSE:
        k = cv2.waitKey(0)
        if k == 112:
            cv2.waitKey()

    if 0xFF & cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()