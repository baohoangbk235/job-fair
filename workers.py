import signal 
import multiprocessing as mp
import pika 
import sys
import cv2
import numpy as np 
from mtcnn.mtcnn import MTCNN
import pickle
import face_recognition
from utils import CentroidTracker
import time
import collections
from datetime import datetime
import csv
import os
import pickle
from db import CheckinManager
import math

import yaml
CONFIG_PATH = 'config.yaml'
try:
    config_file = open(CONFIG_PATH, 'r')
    cfg = yaml.safe_load(config_file)
except:
    raise ("Error: Config file does not exist !")

CPU_COUNT = cfg['cpu_count']
DATABASE = cfg['database']
LABEL_PATH = cfg['label_path']
ENCODING_PATH = cfg['encoding_path']
KNN_MODEL = cfg['trained_model']
RETRAIN_DIR = cfg['retrain_images_dir']

with open(LABEL_PATH,'rb') as f:
	labels = list(np.load(f))

with open(KNN_MODEL,'rb') as f:
	knn = pickle.load(f)

with open(ENCODING_PATH,'rb') as f:
    encodings = list(np.load(f))

retrain_images_dir = '/home/baohoang235/Workplace/face-check-in/retrain_images'

name = None
total_time = 0
total_detect = 0 

c = CheckinManager(DATABASE)

def create_worker():

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))

    channel = connection.channel()

    channel.queue_declare(queue='frame_queue', durable=True)
    print(' [*] Waiting for messages. To exit press CTRL+C')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='frame_queue', on_message_callback=callback)

    channel.start_consuming()

    return callback

def callback(ch, method, properties, body):
    print("\n[INFO] Receiving ...")
    try:
        start = time.time()
        mes = np.frombuffer(body, dtype=np.uint8)
        len_mes = len(mes)/3
        min_side = int(math.sqrt(len_mes + 1)-1)

        mes = mes.reshape((min_side, min_side+2, 3))
        
        frame = mes[:,:min_side,:]

        max_side = mes[0,min_side,0]

        if mes[0,min_side+1,0] == 1:
            frame = cv2.resize(frame, (max_side,min_side))
        else:
            frame = cv2.resize(frame, (min_side,max_side))


        global csv_reader
        global name
        global c 
        global encodings 

        h,w,_ = frame.shape


        faces_encodings = face_recognition.face_encodings(frame, [[0, w, h, 0]])
        results = knn.kneighbors(faces_encodings, n_neighbors=1)
        if results[0][0][0] < 0.6:
            print(labels[int(results[1][0][0])])
        else:
            print("Unknown")
        
        end = time.time()
        print(f'Time processing: {end - start} s.')

        global total_detect 
        global total_time 
        total_detect += 1
        total_time += end-start

    except Exception as e:
        print(str(e))
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return
       
    ch.basic_ack(delivery_tag=method.delivery_tag)
    return 1

def exit_signal_handler(sig, frame):
    print('You pressed Ctrl+C.')
    sys.exit()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_signal_handler)
    ctx = mp.get_context('spawn')
    process_list = []
    for i in range(CPU_COUNT):
        p = ctx.Process(target=create_worker, args=())
        process_list.append(p)
        
    for p in process_list:
        p.start()
    
    for p in process_list:
        p.join()

