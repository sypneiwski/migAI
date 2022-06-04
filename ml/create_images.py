import os
import pickle
import random
import sqlite3

import cv2

def create_db():
    if not os.path.exists("images"):
        os.makedirs("images")
    if not os.path.exists("migai_db.db"):
        conn = sqlite3.connect("migai_db.db")
        conn.execute("CREATE TABLE migai (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
        conn.commit()

def add_gesture(name):
    conn = sqlite3.connect("migai_db.db")
    try:
        conn.execute("INSERT INTO migai (name) VALUES (?)", (name,))
    except sqlite3.IntegrityError:
        print("Gesture already in database")
        return conn.execute("SELECT id FROM migai WHERE name == (?)", (name,)).fetchall()[0][0]
    conn.commit()
    return conn.execute("SELECT id FROM migai WHERE name == (?)", (name,)).fetchall()[0][0]

def take_images(init_pic_no, id):
    pic_no = init_pic_no
    total = pic_no + 25
    started = False
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to capture video")
        return
    while(True):
        frame = cap.read()[1]
        flip = cv2.flip(frame, 1)
        gray = cv2.cvtColor(flip, cv2.COLOR_BGR2GRAY)
        if started:
            cv2.imwrite("images/"+str(id)+"/"+str(pic_no)+".jpg", gray)
            pic_no += 1
        cv2.imshow("image", gray)
        key = cv2.waitKey(1)
        if key == ord('x'):
            if started:
                started = False
                total = init_pic_no
            else:
                print("Started capturing")
                started = True
        if pic_no == total:
            return

create_db()
name = input("Gesture name:...")
init_pic_no = int(input("Initial picture number:..."))
id = add_gesture(name)
if not os.path.exists("images/"+str(id)):
    os.makedirs("images/"+str(id));
print("`x` to start taking images")
take_images(init_pic_no, id)
