import cv2
from os import listdir, path
import json
import pickle
INSPECTION_DIR = path.abspath('inspections/')

import sys
cam_i = int(sys.argv[1])
start_sec = float(sys.argv[2])
end_sec = float(sys.argv[3])

def img_to_time(img):
    return float(img[img.rfind('_') + 1 : img.rfind('.jpg')])

imgs = [f for f in listdir(INSPECTION_DIR) if f.endswith('.jpg') and 'cam_{}'.format(cam_i) in f and start_sec < img_to_time(f) < end_sec]
imgs.sort(key=lambda img : img_to_time(img))
window_name = 'inspection_camera{}'.format(cam_i)
cv2.namedWindow(window_name)


for imgname in imgs:
    print(imgname)
    img = cv2.imread(path.join(INSPECTION_DIR, imgname))
    print('size:', img.size)
    img  = cv2.resize(img, (600, 400))
    # cv2.putText(img, str(img_to_time(imgname)), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    looking_content = None
    touching_content = None
    with open(path.join(INSPECTION_DIR, 'cam_{}_time_{}_looking.p'.format(cam_i, img_to_time(imgname))), 'rb') as f:
        looking_content = pickle.load(f)
    with open(path.join(INSPECTION_DIR, 'cam_{}_time_{}_touching.p'.format(cam_i, img_to_time(imgname))), 'rb') as f:
        touching_content = pickle.load(f)

    print(looking_content)
    print(touching_content)
    def draw_text(lines_and_color, x_pos = 50):
        line_height = 20
        for line_i, (line, color) in enumerate(lines_and_color):
            cv2.putText(img, line, (x_pos, 50 + line_height*line_i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)    
    names = set([x[0] for x in touching_content.keys()])
    lines = [('touching:', (255, 0, 0))]
    import itertools
    for name1, name2 in itertools.product(names, names):
        if name1 is name2 or hash(name1) < hash(name2): # to only iter each pair once
            continue
        touching = touching_content[(name1, name2)]
        lines.append(('{}<->{}:{}'.format(name1, name2, touching), (0, 0, 255) if touching else (50, 50, 50)))
    draw_text(lines)
    lines = []
    lines.append(('looking:', (255, 0, 0)))
    names = set([x[0] for x in looking_content.keys()])

    for name1, name2 in itertools.product(names, names):
        if name1 is name2:
            continue
        looking = looking_content[(name1, name2)]
        looking_dist = -1
        if looking:
            looking, lookint_dist = looking_content[(name1, name2)]
        lines.append(('{}-->{}:{}'.format(name1, name2, looking), (0, 0, 255) if looking else (50, 50, 50)))
    draw_text(lines, x_pos = 400)

        
    # if DRAW_TIME:
        # cv2.putText(img, "time:{:.2f}".format(frame_i / fpss[cap_i]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.imshow(window_name, img)
    # cv2.resizeWindow('some{}'.format(cap_i), 100, 100)

    if 0xFF & cv2.waitKey(50) == 27:
        break
cv2.destroyAllWindows()