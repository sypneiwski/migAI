import cv2
import numpy as np
import os, pickle

def create_histogram():
    have_hist = False
    cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 100, 200
    histogram_avg = np.zeros((180, 256))
    how_many = 0
    print("`x` to take new histogram, `c` to add to average, `z` to exit")
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        crop_img = img[y:y+h, x:x+w]
        if (have_hist):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = cv2.calcBackProject([img], [0, 1], histogram, [0, 180, 0, 256], 1)
            # their code
            dst1 = img.copy()
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(img,-1,disc,img)
            blur = cv2.GaussianBlur(img, (11,11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            img = cv2.merge((thresh,thresh,thresh))
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.imshow("Capturing", img)
        key = cv2.waitKey(1)
        if key == ord('x'):
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([crop_img], [0,1], None, [180,256], [0,180,0,256])
            cv2.normalize(histogram,histogram,0,255,cv2.NORM_MINMAX)
            have_hist = True
        if key == ord('c') and have_hist:
            histogram_avg *= how_many
            histogram_avg += histogram
            how_many += 1
            histogram_avg /= how_many
            print("average over " + str(how_many))
            with open("hand_histogram", "wb") as file:
                pickle.dump(histogram, file)
        if (key == ord('z')):
            return

create_histogram()        
