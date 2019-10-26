import pika 
import cv2 
import numpy as np 
import threading
# from mtcnn.mtcnn import MTCNN
import face_recognition
import yaml
import argparse
import pickle
from centroid_tracker import CentroidTracker
from fps import FPS, WebcamVideoStream
import time 
from facerecognitor import KNNClassifier, utils
import os 

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cam", type=int, default=0,
help="camera source")
ap.add_argument("-i", "--img", type=str, default=None,
help="image path")
ap.add_argument("-t", "--type", type=str, default='video',
help="")
ap.add_argument("-m", "--mode", type=str, default='test',
help="training vs testing mode")
args = vars(ap.parse_args())


CONFIG_PATH = 'config.yaml'
try:
    config_file = open(CONFIG_PATH, 'r')
    cfg = yaml.safe_load(config_file)
except:
    raise ("Error: Config file does not exist !")
CPU_COUNT = cfg['cpu_count']
DATABASE = cfg['database']
model_dir = cfg['model_dir']
LABEL_PATH = cfg['label_path']
ENCODING_PATH = cfg['encoding_path']
KNN_MODEL = cfg['trained_model']
RETRAIN_DIR = cfg['retrain_images_dir']
DOWNSCALE = cfg['down_scale']
train_dir = cfg['train_dir']

font = cv2.FONT_HERSHEY_SIMPLEX


def send_frame(frame, face_location):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    h,w,_ = frame.shape
    mes = frame.astype(np.uint8).tobytes()

    frame = frame.astype(np.uint8)
    top, right, bottom, left = face_location
    face_frame = frame[top:bottom, left:right]

    min_side = min(bottom-top, right-left)
    max_side = max(bottom-top, right-left)

    face_frame = cv2.resize(face_frame, (min_side, min_side))

    camera = np.zeros((min_side,2,3)) 
    camera[:,0,:] = max_side
    if h == max_side:
        camera[:,1,:] = 1
    else:
        camera[:,1,:] = 2

    mes = np.hstack((face_frame, camera)).astype(np.uint8).tobytes()

    
    channel.basic_publish(exchange='',
        routing_key='frame_queue',
        body=mes,
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        ))

    connection.close()

def detect_cam(cam=0):
    vs = WebcamVideoStream(src=cam).start()
    fps = FPS().start()
    count_detect = 0
    detect_delay = 4

    ct = CentroidTracker()

    while(True):
        start = time.time()

        frame = vs.read()

        h,w,_ = frame.shape

        rects = []

        ratio = int(w/640)

        resized_frame = cv2.resize(frame, (640,480))

        if frame is None:
            break

        h,w,_ = resized_frame.shape

        print(h,w)

        rgb_frame = resized_frame[:,:,::-1]
        if count_detect%detect_delay == 0:
            st = time.time()
            face_locations = face_recognition.face_locations(rgb_frame)
            print(f"Time: {time.time() - st}")
            if len(face_locations) > 0:
                for face in face_locations:
                    top, right, bottom, left = face
                    cv2.rectangle(frame, (left, top),(right,bottom), (0,255,0), 2)

                    top = int(ratio * top)
                    right = int(ratio * right)
                    bottom = int(ratio * bottom)
                    left = int(ratio * left)

                    faces_encodings = face_recognition.face_encodings(frame, [[top, right, bottom, left]])
                    results = knn.kneighbors(faces_encodings, n_neighbors=1)

                    if results[0][0][0] < 0.6:
                        cv2.putText(resized_frame, labels[int(results[1][0][0])], (left, top - 10), font, 0.5, (0,255,0), 2)
                        rects.append([left, top, right, bottom, labels[int(results[1][0][0])]])

                    else:
                        cv2.putText(resized_frame, "Unknonw", (left, top - 10), font, 0.5, (0,255,0), 2)
                        rects.append([left, top, right, bottom, "Unknown"])

        objects = ct.update(rects)
        names = ct.get_names()
        locations = ct.get_locations()
        for (objectID, _) in objects.items():
            text = names[objectID]
            startX, startY, endX, endY = locations[objectID]
            cv2.putText(resized_frame, text, (startX , startY-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            cv2.rectangle(resized_frame, (startX, startY),(endX,endY), (0,255,0), 2)
            
        
        end = time.time()

        count_detect += 1

        print(f"FPS : {float((end-start))}")

        cv2.imshow('frame', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps.update()

    fps.stop()
    cv2.destroyAllWindows()

    vs.stop()


def detect_img(img_path, knn_clf, labels, ratio=1, threshold=0.6):
    if img_path is None:
        print("Image path must be specified!")
        exit(0)

    frame = cv2.imread(img_path, 1)
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
            results = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

            if results[0][0][0] < threshold:
                cv2.putText(frame, labels[int(results[1][0][0])], (left, top - 10), font, 0.5, (0,255,0), 2)

            else:
                cv2.putText(frame, "Unknonw", (left, top - 10), font, 0.5, (0,255,0), 2)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

TRAINED_KNN_MODEL = os.path.join(model_dir, "trained_knn_model.clf")

def train():
    print("Training KNN classifier...")
    model = KNNClassifier()
    model.train(train_dir, model_save_path=TRAINED_KNN_MODEL, n_neighbors=None, verbose=True)
    print("Training complete!")

if __name__ == "__main__":
    with open(LABEL_PATH,'rb') as f:
        print("Loading labels...")
        labels = list(np.load(f))

    with open(KNN_MODEL,'rb') as f:
        print("Loading KNN classifier...")
        knn = pickle.load(f)
 
    if args["mode"] == 'test':
        if args["type"] == 'video':
            detect_cam(args["cam"])
        else:
            detect_img(args["img"], knn, labels)
    elif args["mode"] == 'train':
        train()
    else:
        print("Mode have to be specified correctly!")
        