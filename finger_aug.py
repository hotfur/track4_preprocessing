import randomFinger as rF
import cv2
import os
from concurrent.futures import ThreadPoolExecutor


segmentation_dir = 'label/'
output_dir = "label_augmented/"
files = os.listdir(segmentation_dir)

def augment(f):
    img = cv2.imread(segmentation_dir + f)
    res = rF.randomFinger(img, p=1.)
    cv2.imwrite(output_dir + f, res)

with ThreadPoolExecutor() as executor:
    for f in files:
        executor.submit(augment, f)