import os
import pickle
import random
import sqlite3
import time

import cv2


def get_histogram():
    with open("general_hand_histogram", "rb") as file:
        histogram = pickle.load(file)
    return histogram

def create_db():
    if not os.path.exists("images"):
        os.makedirs("images")
    if not os.path.exists("migai_db.db"):
        conn = sqlite3.connect("migai_db.db")
        conn.execute("CREATE TABLE migai (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, amount INTEGER NOT NULL)")
        conn.commit()


def add_gesture(name):
    conn = sqlite3.connect("migai_db.db")
    cmd = "INSERT INTO migai (name, amount) VALUES (?, 0)"
    try:
        conn.execute(cmd, (name,))
    except sqlite3.IntegrityError:
        print("Gesture already in database")
        return
    print("Adding gesture to database")
    conn.commit()


def get_id(name):
    conn = sqlite3.connect("migai_db.db")
    return conn.execute("SELECT id FROM migai WHERE name == (?)", (name,)).fetchall()[0][0]


def get_amount(name):
    conn = sqlite3.connect("migai_db.db")
    return conn.execute("SELECT amount FROM migai WHERE name == (?)", (name,)).fetchall()[0][0]


def update_amount(name, new_amount):
    conn = sqlite3.connect("migai_db.db")
    conn.execute("UPDATE migai SET amount == (?) WHERE name == (?)", (new_amount, name,))
    conn.commit()

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

def take_images(init_pic_no, id):
    pic_no = init_pic_no
    total = pic_no + 25 -1
    started = False
    waiting = 120
    histogram = get_histogram()
    ROI_top, ROI_bottom, ROI_right, ROI_left = 100, 300, 350, 550
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to capture video")
        return
    print("Press `x` to start or restart taking images, `z` to return")
    while True:
        frame = cap.read()[1]
        frame = cv2.flip(frame, 1)
        roi = frame.copy()[ROI_top:ROI_bottom, ROI_right:ROI_left]
        transformed = transform(roi, histogram)
        cv2.rectangle(frame, (ROI_left, ROI_top), (ROI_right,ROI_bottom), (255,0,0), 2)
        if transformed is not None:
            transformed_img, contour = transformed
            cv2.drawContours(frame, [contour + (ROI_right,ROI_top)], -1, (255, 0, 0),1)
            cv2.imshow("transformed", transformed_img)
            if started:
                cv2.putText(frame, "Capturing...", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,), 2)
                if waiting == 0:
                    cv2.imwrite("images/" + str(id) + "/" + str(pic_no) + ".jpg", transformed_img)
                    pic_no += 1
                else:
                    waiting -= 1
        cv2.imshow("image", frame)
        key = cv2.waitKey(1)
        if key == ord('x'):
            if started:
                started = False
                waiting = 50
                total = init_pic_no
            else:
                started = True
        if pic_no == total:
            update_amount(name, total)
            return
        if key == ord('z'):
            return


create_db()
name = input("Gesture name:...")
add_gesture(name)
id = get_id(name)
init_pic_no = get_amount(name) + 1

if not os.path.exists("images/" + str(id)):
    os.makedirs("images/" + str(id));

take_images(init_pic_no, id)

