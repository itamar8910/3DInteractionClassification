import pickle

from PIL import Image

from face_recog.face_recognition_knn import train, predict

def run_train(faces_dir, save_dir):
    train(faces_dir, model_save_path=save_dir, n_neighbors=2)

knn_save_path = 'faces/models/knn2.p'
knn_model = None

def run_predict(img_path):
    global knn_model
    if knn_model is None:
        with open(knn_save_path, 'rb') as f:
            knn_model = pickle.load(f)

    return predict(img_path, knn_model)

def predict_from_keypoints(img_path, humans_keypoints):
    "skip the localization step"
    FACE_WIDTH = 100
    img = Image.open(img_path)
    for keypoints in humans_keypoints:
        nose_pos = keypoints['Nose']
        face_img = img.crop((nose_pos['x'] - FACE_WIDTH/2,
                  nose_pos['y'] - FACE_WIDTH/2,
                  nose_pos['x'] + FACE_WIDTH/2,
                  nose_pos['y'] + FACE_WIDTH/2)
                 )
        face_img.save('dummy/face.png')
        print(run_predict('dummy/face.png'))
        exit()


if __name__ == "__main__":
    # run_train('faces/faces', knn_save_path)
    # print('done')
    # print(run_predict('faces/test/alon_test.png'))
    print(run_predict('dummy/scene_1_cam_1.png'))