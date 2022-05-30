from typing import Tuple, List
from os import listdir
from os.path import isfile, join
import natsort

import numpy as np
import cv2
from label import Label

from augmentation import augment


def getFilesInDir(folder: str, ext: str) -> List[str]:
    files = [join(folder, f) for f in listdir(folder) if (isfile(join(folder, f)) and f.endswith(ext))]
    return files


def loadLabels(shape: Tuple[int, int], folder: str) -> [Label]:
    # Load label file names
    file_labels = natsort.natsorted(getFilesInDir(folder, ".txt"), reverse=True)
    labels = []
    for file in file_labels:
        type_vals = []
        x_vals = []
        y_vals = []
        width_vals = []
        height_vals = []
        for line in open(file).readlines():
            numbers = line.split()

            if len(numbers) != 5:
                break

            obj_type, x, y, width, height = [float(num) for num in numbers[0:5]]
            type_vals.append(obj_type)
            x_vals.append(x)
            y_vals.append(y)
            width_vals.append(width)
            height_vals.append(height)

        labels.append(Label(shape, type_vals, x_vals, y_vals, width_vals, height_vals))

    return labels


def saveLabels(folder: str, basename: str, labels: List[Label]):
    i = 1
    num_digits = len(str(len(labels)))
    for label in labels:
        file_name = f"{basename}{i:0{num_digits}d}.txt"
        with open(join(folder, file_name), "w") as f:
            f.writelines(label.toStr())
        i += 1


def saveImages(ext: str, folder: str, basename: str, images: List[np.array]):
    i = 1
    num_digits = len(str(len(images)))
    for img in images:
        file_name = f"{basename}{i:0{num_digits}d}{ext}"
        cv2.imwrite(join(folder, file_name), img)
        i += 1


def loadImages(folder: str) -> List[np.array]:
    images = natsort.natsorted(getFilesInDir(folder, ".jpg"), reverse=True)
    return [cv2.imread(img) for img in images]


boxes = loadLabels((1, 1), "/home/biblbrox/Projects/Prometei/nn/dataset/labeled/")
images = loadImages("/home/biblbrox/Projects/Prometei/nn/dataset/labeled/")
aug_images, aug_labels = augment(images, boxes)
saveLabels("/home/biblbrox/Projects/Prometei/nn/dataset/augmented", "frame", aug_labels)
saveImages(".jpg", "/home/biblbrox/Projects/Prometei/nn/dataset/augmented", "frame", aug_images)
#cv2.imshow("Window", aug_images[0])
#cv2.waitKey(-1)

