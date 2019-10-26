import os
import cv2
import yaml
import pickle
import argparse
import numpy as np
import face_recognition
import sqlite3

from pathlib import Path
from utils import str2float
from collections import Counter
from mtcnn.mtcnn import MTCNN
from facerecognitor import KNNClassifier, utils

CONFIG_PATH = 'config.yaml'
try:
    config_file = open(CONFIG_PATH, 'r')
    cfg = yaml.safe_load(config_file)
except:
    raise ("Error: Config file does not exist !")

train_dir = cfg['train_dir']
test_dir = cfg['test_dir']
model_dir = cfg['model_dir']
n_neighbors = cfg['n_neighbors']
label_path = cfg['label_path']
face_thresh = cfg['face_thresh']

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train',
                    help='running mode [train, test]')

parser.add_argument('--path', type=str, default=None,
                    help='image path')

args = parser.parse_args()

TRAINED_KNN_MODEL = os.path.join(model_dir, "trained_knn_model.clf")

def train():
    print("Training KNN classifier...")
    model = KNNClassifier()
    classifier = model.train(train_dir, model_save_path=TRAINED_KNN_MODEL, n_neighbors=None, verbose=True)
    print("Training complete!")
    
def test():
    print("Loading KNN classifier...")
    with open(TRAINED_KNN_MODEL, 'rb') as f:
        model = pickle.load(f)

    labels = np.load(label_path)
    print("Loading complete")

    acc = {}
    # STEP 2: Using the trained classifier, make predictions for unknown images
    for person in os.listdir(train_dir):
        images = os.listdir(os.path.join(train_dir, person))
        if len(images) == 0:
            continue

        print("Processing {} [{} images] ..".format(person, len(images)))

        if len(images) > 1:
            sorted_images = sorted(images, key=lambda x: str2float(os.path.basename(x).split('.png')[0]))[1:]
        else:
            sorted_images = sorted(images, key=lambda x: str2float(os.path.basename(x).split('.png')[0]))
    
        acc[person] = {}
        acc[person]['num_images'] = len(images)
        acc[person]['top1'] = []
        acc[person]['top5'] = []
        acc[person]['num_checkins'] = counter[person]
        acc[person]['distance_top1'] = []
        acc[person]['distance_top5'] = []

        all_predictions = []
        for idx, im in enumerate(sorted_images):
            print("{}/{}".format(idx + 1, len(images)), end='\r', flush=True)
            full_file_path = os.path.join(train_dir, person).replace("/", r"\\") + r"\\" + im
            # X_img = cv2.imread(full_file_path)[:, :, ::-1]
            X_img = face_recognition.load_image_file(full_file_path, mode='RGB')
            if len(images) <= 1:
                X_face_locations = face_recognition.face_locations(X_img)
            else:
                h, w = X_img.shape[:2]
                X_face_locations = [[0, w, h, 0]]

            # Find encodings for faces in the test image
            faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

            # Use the KNN model to find the best matches for the test face
            res = model.kneighbors(faces_encodings, n_neighbors=50)
            all_predictions.append(res)
        
        batch_size = 10
        num_batches = len(sorted_images) // batch_size
        for i in range(num_batches + 1):
            low = i
            if i + batch_size >= len(sorted_images):
                hi = len(sorted_images)
                low = max(0, hi - batch_size)
            else:
                hi = low + batch_size

            neighbors = {}
            for res in all_predictions[low:hi]:
                dis = res[0][0].tolist()
                nbs = res[1][0].tolist()
                for d, n in zip(dis, nbs):
                    pred = labels[n]
                    if d < face_thresh:
                        if pred not in neighbors:
                            neighbors[pred] = [d]
                        else:
                            neighbors[pred].append(d)

            final_predictions = []
            for p, v in neighbors.items():
                mean_dist = np.mean(v)
                final_predictions.append([p, len(v), mean_dist])

            final_predictions = sorted(final_predictions, key=lambda x: -x[1])

            if person == final_predictions[0][0]:
                acc[person]['top1'].append(1)
                acc[person]['top5'].append(1)
                acc[person]['distance_top1'].append(final_predictions[0][-1])
            elif person in [fp[0] for fp in final_predictions[:5]]:
                acc[person]['top1'].append(0)
                acc[person]['top5'].append(1)
                found = [fp[0] for fp in final_predictions[:5]].index(person)
                acc[person]['distance_top5'].append(final_predictions[found][-1])
            else:
                acc[person]['top1'].append(0)
                acc[person]['top5'].append(0)

        for k in ['top1', 'top5', 'distance_top1', 'distance_top5']:
            if np.sum(acc[person][k]) > 0:
                acc[person][k] = np.mean(acc[person][k])
            else:
                acc[person][k] = 0
        
        print(person, acc[person])
        
    with open("evaluation.pkl", 'wb') as f:
        pickle.dump(acc, f, pickle.HIGHEST_PROTOCOL)

def test_image(image_path):
    print("Loading KNN classifier...")
    model = KNNClassifier(TRAINED_KNN_MODEL, label_path)
    frame = cv2.imread(image_path,1)
    model.my_predict(frame)

if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    if args.mode == 'train':
        train()
    # STEP 1.5: Loaded trained model
    elif args.mode == 'img':
        test_image(args.path)
    else:
        test()
    
