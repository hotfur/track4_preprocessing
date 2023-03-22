import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

src_path = '../dataset/classify/bottles/seg/'
label_path = '../dataset/classify/bottles/label/'
save_path = '../dataset/classify/bottles/filtering/'

files = os.listdir(src_path)
def filter_bottom(f):
    src_img = cv2.imread(src_path+ f)
    label = cv2.imread(label_path + f)

    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    _, label = cv2.threshold(label, 128, 1, cv2.THRESH_BINARY)
    mask_pixels = cv2.countNonZero(label)

    # Calculate radius and area of circle
    rows, cols = label.shape[:2]
    r = min(rows, cols)/2
    circle_area = np.pi * r**2

    # Compute ratio of mask pixels to circle area
    ratio = mask_pixels / circle_area

    if 0.95 < ratio < 1.05 and abs(rows-cols) < 1/3*r:
        cv2.imwrite(save_path+'circle/' + f, src_img)
        cv2.imwrite(save_path + 'circle_label/' + f, label)
    else:
        cv2.imwrite(save_path+'no_circle/' + f, src_img)
        cv2.imwrite(save_path + 'no_circle_label/' + f, label)

with ThreadPoolExecutor() as executor:
    for f in files:
        executor.submit(filter_bottom, f)
