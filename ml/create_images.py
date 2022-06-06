import os
import pickle
import random
import sqlite3
import time

import cv2


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

def take_images(init_pic_no, id):
    pic_no = init_pic_no
    total = pic_no + 25
    started = False
    waiting = 50
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to capture video")
        return
    print("Press `x` to start or restart taking images")
    while (True):
        frame = cap.read()[1]
        flip = cv2.flip(frame, 1)
        gray = cv2.cvtColor(flip, cv2.COLOR_BGR2GRAY)
        if started:
            if waiting == 0:
                cv2.imwrite("images/" + str(id) + "/" + str(pic_no) + ".jpg", gray)
                pic_no += 1
            else:
                waiting -= 1
        cv2.imshow("image", gray)
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


create_db()
name = input("Gesture name:...")
add_gesture(name)
id = get_id(name)
init_pic_no = get_amount(name)

if not os.path.exists("images/" + str(id)):
    os.makedirs("images/" + str(id));

take_images(init_pic_no, id)

total = get_amount(name)

print("You now have ", total, " images of gesture ", name)
