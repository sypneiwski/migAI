import cv2
import numpy as np
import os, pickle

background = None
accumulated_weight = 0.5

#Creating the dimensions for the ROI...
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def cal_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)
    _ , thresholded = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    cv2.filter2D(thresholded,-1,disc,thresholded)
    thresholded = cv2.GaussianBlur(thresholded, (3,3), 0)
    thresholded = cv2.medianBlur(thresholded, 3)
    return thresholded

def background_subtraction():
    cam = cv2.VideoCapture(0)
    num_frames = 0
    started = False

    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img_copy = img.copy()
        roi = img[ROI_top:ROI_bottom, ROI_right:ROI_left]
        cv2.rectangle(img_copy, (ROI_left, ROI_top), (ROI_right,ROI_bottom), (255,0,0), 2)
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        if num_frames < 60 and started:
            cal_accum_avg(gray_frame, accumulated_weight)
            if num_frames <= 59:
                cv2.putText(img_copy, "FETCHING BACKGROUND...PLEASE WAIT",
                        (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        elif started:
            thresholded = segment_hand(gray_frame)
            cv2.imshow("thresholded", thresholded)
        cv2.imshow("capturing", img_copy)
        if started:
            num_frames += 1
        print(num_frames)
        key = cv2.waitKey(1)
        if key == ord('z'):
            return
        elif key == ord('c'):
            started = True

background_subtraction()