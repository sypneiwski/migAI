import cv2
import numpy as np
import os, pickle

def create_histogram():
    have_hist = False
    cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 100, 200
    print("`c` to take new histogram and add to average, 'x' to reset, `z` to exit")
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        crop_img = img[y:y+h, x:x+w]
        if (have_hist):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = cv2.calcBackProject([img], [0, 1], histogram_avg, [0, 180, 0, 256], 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(img,-1,kernel,img)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.imshow("Capturing", img)
        key = cv2.waitKey(1)
        if key == ord('c'):
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([crop_img], [0,1], None, [180,256], [0,180,0,256])
            cv2.normalize(histogram,histogram,0,255,cv2.NORM_MINMAX)
            if not have_hist:
                have_hist = True
                histogram_avg = histogram
                how_many = 1
            else:
                histogram_avg *= how_many
                histogram_avg += histogram
                how_many += 1
                histogram_avg /= how_many
            print("average over " + str(how_many))
            with open("hand_histogram", "wb") as file:
                pickle.dump(histogram_avg, file)
        if key == ord('x'):
            have_hist = False
        if key == ord('z'):
            return

create_histogram()        
