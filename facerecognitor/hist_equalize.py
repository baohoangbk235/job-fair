import numpy as np 
import cv2 

img = cv2.imread('test.jpg', 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

color = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)

cv2.imshow('img', img)

cv2.imshow('final', color)


