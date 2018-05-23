import math

from PIL import Image, ImageDraw, ImageFont


from face_recog.recognize import run_predict
from openpose_util.run_image_get_humans import get_humans_keypoints


def keypoints_to_center(kepoints):
    # assert not all([x is None for x in kepoints.values()])
    return avg([kp['x'] for kp in kepoints.values() if kp is not None]), avg(
        [kp['y'] for kp in kepoints.values() if kp is not None])

def avg(l):
    return sum(l) / len(l)

def get_identities_and_keypoints(img_path):
    """
    returns list of dicts: 'identity': ..., 'keypoints': ...
    :param img_path:
    :return:
    """



    def l2_dis(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p2[1])

    identities = run_predict(img_path)
    humans_keypoints = get_humans_keypoints(img_path)

    # import pickle
    # # pickle.dump(identities, open('dummy/identities.p', 'wb'))
    # # pickle.dump(humans_keypoints, open('dummy/humans_keypoints.p', 'wb'))
    #
    # identities = pickle.load(open('dummy/identities.p', 'rb'))
    # humans_keypoints = pickle.load(open('dummy/humans_keypoints.p', 'rb'))

    humans_keypoints = [kp for kp in humans_keypoints if not all([x is None for x in kp.values()])]
    print(identities)
    print(humans_keypoints)

    identity_and_keypoints = []
    for identity in identities:
        item = {}
        item['identity'] = identity[0]
        face_bbox = identity[1]
        face_center_x = avg([face_bbox[1], face_bbox[3]])
        face_center_y = avg([face_bbox[0], face_bbox[2]])

        closest_keypoints = min(humans_keypoints, key=lambda keypoints : l2_dis((face_center_x, face_center_y), keypoints_to_center(keypoints)))
        item['keypoints'] = closest_keypoints
        identity_and_keypoints.append(item)
    print(identity_and_keypoints)
    return identity_and_keypoints


def draw_humans_and_keypoints(img_path):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    identities_keypoints = get_identities_and_keypoints(img_path)
    for x in identities_keypoints:
        keypoints = x['keypoints']
        identity = x['identity']
        print(identity)
        print(keypoints)
        for part in keypoints.keys():
            if keypoints[part] is not None:
                x = int(keypoints[part]['x'])
                y = int(keypoints[part]['y'])
                draw.rectangle([(x-10, y-10), (x + 10, y + 10)])
                draw.text([(x, y)], part)
            else:
                print(part, ' is missing')
        keypoints_center = keypoints_to_center(keypoints)
        draw.text([(keypoints_center[0], keypoints_center[1] - 40)], '' + identity,  fill=(255,0,0,255))

        # make a blank image for the text, initialized to transparent text color
        # base = Image.open('Pillow/Tests/images/lena.png').convert('RGBA')
        # txt = Image.new('RGBA', base.size, (255, 255, 255, 0))
        #
        # # get a font
        # fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
        # # get a drawing context
        # d = ImageDraw.Draw(txt)
        # d.text((10, 10), "Hello", font=fnt, fill=(255, 255, 255, 128))

    img.show()

if __name__ == "__main__":
    draw_humans_and_keypoints('dummy/test4.png')