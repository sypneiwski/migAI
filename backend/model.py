import torch
import numpy as np
import cv2
import pickle
import sqlite3
from define_model import ConvNet
from PIL import Image
from torchvision.transforms import ToTensor


def get_histogram():
    with open("../ml/general_hand_histogram", "rb") as file:
        histogram = pickle.load(file)
    return histogram

def get_descriptions():
    with open("idx_to_class", "rb") as file:
        idx_to_class = pickle.load(file)
    return idx_to_class


def transform(img, histogram):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.calcBackProject([img], [0, 1], histogram, [0, 180, 0, 256], 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    cv2.filter2D(img,-1,kernel,img)
    img = cv2.GaussianBlur(img, (9,9), 0)
    img = cv2.medianBlur(img, 7)
    img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return img


def predict(img):
    # Get transformed image
    histogram = get_histogram()
    cv2.imwrite("./test.jpg", img)
    transformed = transform(img, histogram)
    #cv2.imwrite("./test.jpg", transformed)

    # Get model
    model = ConvNet()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Get class descriptions
    idx_to_class = get_descriptions()

    # Get database
    conn = sqlite3.connect("../ml/migai_db.db")

    # # Get tensor from image
    # tensor = ToTensor(Image.open("test.jpg").convert("RGB"))
    # input = torch.unsqueeze(img, 0)

    # # Get prediction
    # output = model(input)
    # _, predicted = torch.max(output.data, 1)
    # prediction = conn.execute("SELECT name FROM migai WHERE id == (?)", (idx_to_class[predicted.item()],)).fetchall()[0]
    # print(type(prediction))
    # print("prediction : " + str(prediction))
    # return prediction
    return "X"

