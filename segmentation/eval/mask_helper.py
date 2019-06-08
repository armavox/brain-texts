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


def fixmask(mask):
    # mask = mask / (mask.max() / 255)
    mask = mask.astype(np.uint8)
    _, mask = cv2.threshold(mask, 0 , 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = floodfill(mask)
    mask = remgar(mask)
    mask = mask.astype(np.uint8)
    left = mask[:,:mask.shape[1]//2]
    right = mask[:,mask.shape[1]//2:]
    lsum = cv2.countNonZero(left)
    rsum = cv2.countNonZero(right)
    if lsum == 0 or rsum == 0:
        print('Oh no!')
        return mask
    else:
        if max(lsum/rsum, rsum/lsum) < 1.05:
            return mask
        else:
            rl = right[:,::-1]
            lr = left[:,::-1]
            if rsum > lsum:
                d = rl - left
            else:
                d = lr - right
            opening = cv2.morphologyEx(d, cv2.MORPH_OPEN, kernel)
            if rsum > lsum:
                fixed = left | opening
            else:
                fixed = right | opening
            fixed = floodfill(fixed)
            newmask = mask.copy()
            if rsum > lsum:
                newmask[:,:mask.shape[0]//2] = fixed
            else:
                newmask[:,mask.shape[0]//2:] = fixed
            newmask = cv2.morphologyEx(newmask, cv2.MORPH_CLOSE, kernel)
            newmask = floodfill(newmask)
            return newmask
