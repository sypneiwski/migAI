import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
from PIL import Image
import sqlite3


def get_histogram():
    with open("general_hand_histogram", "rb") as file:
        histogram = pickle.load(file)
    return histogram

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(16, 32, 6)
        # self.fc1 = nn.Linear(32*21*21, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 22)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 6)
        self.fc1 = nn.Linear(16*21*21, 22)

    def forward(self, x):
        # Convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flattening
        x = torch.flatten(x, 1)
        # Fully connected layers
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.fc1(x)
        return x

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
    model = ConvNet()
    conn = sqlite3.connect("migai_db.db")
    model.load_state_dict(torch.load("model.pth"))
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
            cv2.imwrite("prediction/1/test.jpg", transformed_img)
            test = datasets.ImageFolder("prediction/", transform=ToTensor())
            xxx = datasets.ImageFolder("val/", transform=ToTensor())
            sample_idx = torch.randint(len(test), size=(1,)).item()
            img, label = test[sample_idx]
            input = torch.unsqueeze(img, 0)

            output = model(input)
            _, predicted = torch.max(output.data, 1)
            idx_to_class = {v: k for k, v in xxx.class_to_idx.items()}
            prediction = conn.execute("SELECT name FROM migai WHERE id == (?)", (idx_to_class[predicted.item()],)).fetchall()[0]
            print(prediction)

        if key == ord('z'):
            return

take_images()
