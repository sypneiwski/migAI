import cv2
import pickle
from sklearn.utils import shuffle
from glob import glob
import numpy as np
import os


def get_data():
    labels = []
    images = []
    files = glob("images/*/*.jpg")
    for file in files:
        label = file[file.find(os.sep) + 1: file.rfind(os.sep)]
        labels.append(label)
        image = cv2.imread(file, 0)
        images.append(np.array(image))
    return labels, images


def dump(file_name, data):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_data():
    labels, images = get_data()
    print(len(labels), ", ", len(images), " = size of labels, size of images")
    labeled_images = list(zip(labels, images))
    labeled_images = shuffle(labeled_images)
    labels, images = zip(*labeled_images)
    print(len(labels), ", ", len(images), " = size of labels, size of images")
    size = len(images)
    img_train = images[:int(8 / 10 * size)]
    img_test = images[int(8 / 10 * size):int(9 / 10 * size)]
    img_val = images[int(9 / 10 * size):]
    lab_train = labels[:int(8 / 10 * size)]
    lab_test = labels[int(8 / 10 * size):int(9 / 10 * size)]
    lab_val = labels[int(9 / 10 * size):]

    print(len(img_train), len(img_test), len(img_val), len(lab_train), len(lab_test), len(lab_val))
    dump("images_train", img_train)
    dump("images_test", img_test)
    dump("images_validate", img_val)
    dump("labels_train", lab_train)
    dump("labels_test", lab_test)
    dump("labels_validate", lab_val)


load_data()
