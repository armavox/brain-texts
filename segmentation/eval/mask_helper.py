import cv2
import numpy as np


def floodfill(img):
    f = img.copy()
    f = cv2.floodFill(f,None,(0,0),255)
    x = ~f[1] | img
    return x


def remgar(image): #remove garbage
    image = image.astype(np.uint8)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(image, connectivity = 8)
    sizes = stats[:, -1]
    max_label = 0
    max_size = sizes[0]
    if len(sizes)>1:
        max_label = 1
        max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    cleanimage = np.zeros(output.shape)
    cleanimage[output == max_label] = 255
    return cleanimage


def fixmask(m):
    m = m / (m.max() / 255)
    m = m.astype(np.uint8)
    _, thr = cv2.threshold(m, 0 , 255, cv2.THRESH_OTSU)
    m = remgar(thr)
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(75,75)))
    m = floodfill(m)
    return m
