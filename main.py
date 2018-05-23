import cv2

captures = [
    cv2.VideoCapture('footage/footage_synched/footage1.mp4'),
    cv2.VideoCapture('footage/footage_synched/footage2.mp4'),
    cv2.VideoCapture('footage/footage_synched/footage3.mp4'),
    cv2.VideoCapture('footage/footage_synched/footage4.mp4'),

]

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.

fpss = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
print(fpss)

PAUSE = True
DRAW_TIME = True

frame_i = 0
while True:
    for cap_i, cap in enumerate(captures):
        ret, img = cap.read()
        img  = cv2.resize(img, (600, 400))
        if DRAW_TIME:
            cv2.putText(img, "time:{:.2f}".format(frame_i / fpss[cap_i]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow('some{}'.format(cap_i), img)
        # cv2.resizeWindow('some{}'.format(cap_i), 100, 100)
    frame_i += 1

    if PAUSE:
        k = cv2.waitKey(0)
        if k == 112:
            cv2.waitKey()

    if 0xFF & cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()