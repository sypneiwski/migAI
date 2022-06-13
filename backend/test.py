from define_model import ConvNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2
import os
import pickle
from PIL import Image
import sqlite3



model = ConvNet()
model.load_state_dict(torch.load("model.pth"))
with open("idx_to_class", "rb") as file:
    idx_to_class = pickle.load(file)


def get_histogram():
    with open("../ml/hand_histogram", "rb") as file:
        histogram = pickle.load(file)
    return histogram



def transform(img, histogram):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.calcBackProject([img], [0, 1], histogram, [0, 180, 0, 256], 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    cv2.filter2D(img,-1,kernel,img)
    img = cv2.GaussianBlur(img, (9,9), 0)
    img = cv2.medianBlur(img, 7)
    img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    if len(contours) == 0:
        return None
    else:
        max_contour = max(contours, key=cv2.contourArea)
        return (img, max_contour)

def take_images():
    histogram = get_histogram()
    conn = sqlite3.connect("../ml/migai_db.db")
    model.eval()
    ROI_top, ROI_bottom, ROI_right, ROI_left = 100, 300, 350, 550
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to capture video")
        return
    print("Press `x` to take image, `z` to return")
    while True:
        frame = cap.read()[1]
        frame = cv2.flip(frame, 1)
        roi = frame.copy()[ROI_top:ROI_bottom, ROI_right:ROI_left]
        transformed = transform(roi, histogram)
        cv2.rectangle(frame, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 0, 0), 2)
        if transformed is not None:
            transformed_img, contour = transformed
            cv2.imshow("transformed", transformed_img)
        cv2.imshow("image", frame)
        key = cv2.waitKey(1)
        if key == ord('x') and transformed is not None:
            transformed_img, _ = transformed
            cv2.imwrite("test.jpg", transformed_img)
            transformation = ToTensor()
            tensor = transformation(Image.open("test.jpg").convert("RGB"))
            input = torch.unsqueeze(tensor, 0)

            output = model(input)
            _, predicted = torch.max(output.data, 1)
            prediction = conn.execute("SELECT name FROM migai WHERE id == (?)", (idx_to_class[predicted.item()],)).fetchall()[0]
            print(prediction)

        if key == ord('z'):
            return

take_images()
