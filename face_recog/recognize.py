import pickle

from face_recog.face_recognition_knn import train, predict

def run_train(faces_dir, save_dir):
    train(faces_dir, model_save_path=save_dir, n_neighbors=2)

knn_save_path = 'faces/models/knn1.p'
knn_model = None

def run_predict(img_path):
    global knn_model
    if knn_model is None:
        with open(knn_save_path, 'rb') as f:
            knn_model = pickle.load(f)

    return predict(img_path, knn_model)

if __name__ == "__main__":
    # run_train('faces/faces', knn_save_path)
    # print('done')
    print(run_predict('faces/test/alon_test.png'))