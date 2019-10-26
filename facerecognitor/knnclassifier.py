import os
import cv2
import math
import pickle 
from datetime import datetime
import numpy as np
import face_recognition
from mtcnn.mtcnn import MTCNN

from sklearn import neighbors
from collections import Counter
from face_recognition.face_recognition_cli import image_files_in_folder

from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                #iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    #rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

import yaml
CONFIG_PATH = 'config.yaml'
try:
    config_file = open(CONFIG_PATH, 'r')
    cfg = yaml.safe_load(config_file)
except:
    raise ("Error: Config file does not exist !")


def str2float(x):
    res = 0.0
    try:
        res = float(x)
    except ValueError:
        res = 0.0
    return res


class KNNClassifier(object):
    """
    K-nearest neighbors classifier for face recognition 
    with the following embedding https://github.com/ageitgey/face_recognition
    """
    def __init__(self, model_path=None, label_path=None):
        if model_path:
            with open(model_path, 'rb') as f:
                self.knn_clf = pickle.load(f)
        
        if label_path:
            self.labels = np.load(label_path).tolist()
        else:
            self.labels = None
        
        self.detector = MTCNN()

    def train(self, train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        """
        Trains a k-nearest neighbors classifier for face recognition.
        :param train_dir: directory that contains a sub-directory for each known person, with its name.
        (View in source code to see train_dir example tree structure)
        Structure:
            <train_dir>/
            ├── <person1>/
            │   ├── <somename1>.jpeg
            │   ├── <somename2>.jpeg
            │   ├── ...
            ├── <person2>/
            │   ├── <somename1>.jpeg
            │   └── <somename2>.jpeg
            └── ...
        :param model_save_path: (optional) path to save model on disk
        :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
        :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
        :param verbose: verbosity of training
        :return: returns knn classifier that was trained on the given data.
        """
        X, y = [], []

        CONFIG_PATH = 'config.yaml'
        try:
            config_file = open(CONFIG_PATH, 'r')
            cfg = yaml.safe_load(config_file)
        except:
            raise ("Error: Config file does not exist !")

        loadingNumpy = False

        # Loop through each person in the training set
        if not loadingNumpy:
            for class_dir in os.listdir(train_dir):

                if not os.path.isdir(os.path.join(train_dir, class_dir)):
                    continue

                print("Gathering data in {}".format(class_dir))
                class_images = image_files_in_folder(os.path.join(train_dir, class_dir))
                # Loop through each training image for the current person

                class_images = sorted(class_images, key=lambda x: str2float(os.path.basename(x).split('.png')[0]))

                for img_path in class_images[-50:]:
                    image = face_recognition.load_image_file(img_path)
                    image = aug_pipe.augment_image(image)

                    face_bounding_boxes = face_recognition.face_locations(image)
    
                    if len(face_bounding_boxes) < 1:
                        # If there are no people (or too many people) in a training image, skip the image.
                        if verbose:
                            print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                    else:
                        # Add face encoding for current image to the training set
                        X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                        y.append(class_dir)


        save_dir = "/".join(model_save_path.split('/')[:-1])
        
        if loadingNumpy:
            print("Loading numpy data ...")
            y = np.load(os.path.join(save_dir, 'labels.npy'))
            X = np.load(os.path.join(save_dir, 'encodings.npy'))
        else:
            np.save(os.path.join(save_dir, "labels"), np.asarray(y))
            np.save(os.path.join(save_dir, "encodings"), np.asarray(X))

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
            from shutil import copyfile
            copyfile(model_save_path, os.path
                .join(*model_save_path.split("/")[:-1], "{}.clf".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))

        self.knn_clf = knn_clf

    def predict_image(self, X_img, distance_threshold=0.4, scale=1.0, cropped=False):
        if not cropped:
            if scale < 1.0:
                X_img = cv2.resize(X_img, (0, 0), fx=0.25, fy=0.25)
                
            X_face_locations = face_recognition.face_locations(X_img)
            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                return []
        else:
            h, w = X_img.shape[:2]
            X_face_locations = [[0, w, h, 0]]

        # Find encodings for faces in the test image
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

    def predict_prob(self, X_img, distance_threshold=0.4, scale=1.0, k=10, cropped=False):
        assert self.labels is not None
        
        if not cropped:
            if scale < 1.0:
                X_img = cv2.resize(X_img, (0, 0), fx=0.25, fy=0.25)
            
            X_face_locations = face_recognition.face_locations(X_img)
            # X_face_locations = []
            # detected_faces = self.detector.detect_faces(X_img)
            # for face in detected_faces:
            #     x,y,width,height = face["box"]
            #     X_face_locations.append([y,x+width,y+height,x])
            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                return []
        else:
            h, w = X_img.shape[:2]
            X_face_locations = [[0, w, h, 0]]
            
        max_size = -1
        for (top, right, bottom, left) in X_face_locations: 
            size = (bottom - top) * (right - left)     
            if size > max_size:
                max_face = (top, right, bottom, left)
        
        X_face_locations = [max_face]

        # Find encodings for faces in the test image
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        res = self.knn_clf.kneighbors(faces_encodings, n_neighbors=k)

        neighbors = {}
        dis = res[0][0].tolist()
        nbs = res[1][0].tolist()
        for d, n in zip(dis, nbs):
            if d < distance_threshold:
                pred = self.labels[n]
                # pred = nbs
                if pred not in neighbors:
                    neighbors[pred] = [d]
                else:
                    neighbors[pred].append(d)

        people = {}
        # count = []
        # distances = []
        for p, v in neighbors.items():
            people[p] = [len(v), np.mean(v)]

        # distances = np.array(distances, dtype=np.float32)
        # count = np.array(count, dtype=np.float32)

        # distances_prob = np.exp(distances) / np.sum(np.exp(distances), axis=0)
        # count_prob = np.exp(count) / np.sum(np.exp(count), axis=0)

        # final_prob = (distances_prob + count_prob) / 2
        
        # result = list(zip(people, final_prob.tolist()))
        # result = list(zip(people, count))
        # result.sort(key=lambda x: -x[1])

        return [max_face, people]
        
    def predict(self, X_img_path, distance_threshold=0.6, scale=1.0, cropped=False):
        """
        Recognizes faces in given image using a trained KNN classifier
        :param X_img_path: path to image to be recognized
        :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
        :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
        :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
            of mis-classifying an unknown person as a known one.
        :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
            For faces of unrecognized persons, the name 'unknown' will be returned.
        """
        if not os.path.isfile(X_img_path):
            raise Exception("Invalid image path: {}".format(X_img_path))

        # Load image file and find face locations
        X_img = face_recognition.load_image_file(X_img_path)
        return self.predict_image(X_img, distance_threshold, scale, cropped)
    
    def my_predict(self, frame, ratio=1, threshold=0.6):
        import time
        font = cv2.FONT_HERSHEY_SIMPLEX
        rgb_frame = frame[:,:,::-1]
        st = time.time()
        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"Time: {time.time() - st}")
        if len(face_locations) > 0:
            for face in face_locations:
                top, right, bottom, left = face
                top = int(ratio * top)
                right = int(ratio * right)
                bottom = int(ratio * bottom)
                left = int(ratio * left)
                cv2.rectangle(frame, (left, top),(right,bottom), (0,255,0), 2)


                faces_encodings = face_recognition.face_encodings(frame, [[top, right, bottom, left]])
                results = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)

                if results[0][0][0] < threshold:
                    cv2.putText(frame, self.labels[int(results[1][0][0])], (left, top - 10), font, 0.5, (0,255,0), 2)

                else:
                    cv2.putText(frame, "Unknonw", (left, top - 10), font, 0.5, (0,255,0), 2)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


