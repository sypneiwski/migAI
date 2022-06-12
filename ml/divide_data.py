import os
import cv2
from glob import glob
from sklearn.utils import shuffle
import sqlite3

def move(destination, files):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for file in files:
        img_name = file[file.rfind(os.sep) + 1:]
        img = cv2.imread(file, 0)
        cv2.imwrite(destination + "/" + img_name, img)


def divide(id):
    files = glob("images/" + str(id) + "/*.jpg")
    files = shuffle(shuffle(files))
    f_train = files[:int(len(files) * 8 / 10)]
    f_test = files[int(len(files) * 8 / 10): int(len(files) * 9 / 10)]
    f_val = files[int(len(files) * 9 / 10):]
    return f_train, f_test, f_val


if not os.path.exists("train"):
    os.makedirs("train")

if not os.path.exists("test"):
    os.makedirs("test")

if not os.path.exists("validate"):
    os.makedirs("validate")

conn = sqlite3.connect("migai_db.db")
ids = conn.execute("SELECT id FROM migai").fetchall()

for id in ids:
    id = id[0]
    f_train, f_test, f_val = divide(id)
    move("train/" + str(id), f_train)
    move("test/" + str(id), f_test)
    move("val/" + str(id), f_val)


