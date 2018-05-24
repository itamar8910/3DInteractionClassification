import cv2

img = None
def get_click_bgr(event, x, y, flags, args):
    # print('mouse event')
    if event == cv2.EVENT_LBUTTONDOWN:

        print('mouse click:{},{}'.format(x, y))
        print('rgb {}'.format(img[y, x]))


cv2.namedWindow('window')
cv2.setMouseCallback('window', get_click_bgr, param=None)

img = cv2.imread('calibration/frames/cam0/frame_38.100.jpg')
cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()