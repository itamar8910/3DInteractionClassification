import math

from PIL import Image, ImageDraw, ImageFont


from face_recog.recognize import run_predict, predict_from_keypoints
from openpose_util.run_image_get_humans import get_humans_keypoints


def keypoints_to_center(kepoints):
    # assert not all([x is None for x in kepoints.values()])
    return avg([kp['x'] for kp in kepoints.values() if kp is not None]), avg(
        [kp['y'] for kp in kepoints.values() if kp is not None])

def avg(l):
    return sum(l) / len(l)

def get_identities_and_keypoints(img_path):
    """
    for each identity in the image, returns the identity with its keypoints
    returns list of dicts: 'identity': ..., 'keypoints': ...
    :param img_path:
    :return:
    """



    def l2_dis(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p2[1])

    humans_keypoints = get_humans_keypoints(img_path)
    print(humans_keypoints)
    identities = predict_from_keypoints(img_path, humans_keypoints)
    # identities = run_predict(img_path)

    # import pickle
    # # pickle.dump(identities, open('dummy/identities.p', 'wb'))
    # # pickle.dump(humans_keypoints, open('dummy/humans_keypoints.p', 'wb'))
    #
    # identities = pickle.load(open('dummy/identities.p', 'rb'))
    # humans_keypoints = pickle.load(open('dummy/humans_keypoints.p', 'rb'))


    # we have the identities & humans keypoints, we now need to match them

    humans_keypoints = [kp for kp in humans_keypoints if not all([x is None for x in kp.values()])]
    print(identities)
    print(humans_keypoints)

    identity_and_keypoints = []
    identified_humans_indices = []
    for identity in identities:

        if len(humans_keypoints) == 0:
            return

        item = {}
        item['identity'] = identity[0]
        face_bbox = identity[1]
        face_center_x = avg([face_bbox[1], face_bbox[3]])
        face_center_y = avg([face_bbox[0], face_bbox[2]])

        closest_keypoints_i = min(range(len(humans_keypoints)), key=lambda keypoints_i : l2_dis((face_center_x, face_center_y), keypoints_to_center(humans_keypoints[keypoints_i])))
        item['keypoints'] = humans_keypoints[closest_keypoints_i]
        identity_and_keypoints.append(item)
        identified_humans_indices.append(closest_keypoints_i)
        humans_keypoints = [x for i, x in enumerate(humans_keypoints) if i != closest_keypoints_i]

    for left_keypoints in humans_keypoints:
        identity_and_keypoints.append({'identity': 'unkown', 'keypoints': left_keypoints})
    print(identity_and_keypoints)
    return identity_and_keypoints


def draw_humans_and_keypoints(img_path):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    identities_keypoints = get_identities_and_keypoints(img_path)
    print('***')
    print(identities_keypoints)
    print('***')
    for x in identities_keypoints:
        keypoints = x['keypoints']
        identity = x['identity']
        print(identity)
        print(keypoints)
        for part in keypoints.keys():
            if keypoints[part] is not None:
                x = int(keypoints[part]['x'])
                y = int(keypoints[part]['y'])
                draw.rectangle([(x-5, y-5), (x + 5, y + 5)])
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
    draw_humans_and_keypoints('calibration/frames/cam0/frame_25.933.jpg')