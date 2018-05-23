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

frame_i = 0
while True:
    for cap_i, cap in enumerate(captures):
        ret, img = cap.read()
        img  = cv2.resize(img, (600, 400))
        cv2.putText(img, "time:{:.2f}".format(frame_i / fpss[cap_i]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow('some{}'.format(cap_i), img)
        # cv2.resizeWindow('some{}'.format(cap_i), 100, 100)
    frame_i += 1
    # ret1, img1 = capture1.read()
    # ret2, img2 = capture2.read()
    # img1.resize(300,300)
    # img2.resize(300,300)
    # frame_i  += 1
    # # result = processFrame(img)
    #
    # cv2.putText(img1,"time:{:.2f}".format(frame_i / fps1), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    # cv2.imshow('some1', img1)
    #
    # cv2.putText(img2, "time:{:.2f}".format(frame_i / fps2), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    # cv2.imshow('some2', img2)

    # k = cv2.waitKey(0)
    # if k == 112:
    #     cv2.waitKey()

    if 0xFF & cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()