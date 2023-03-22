import rectangle as r
import randomFinger as rF
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
import augmentation as a
A = r.Point(0, 9)
B = r.Point(12, 9)
C = r.Point(0, 0)
print(A.y)
print(r.dist(B, C))
# A = r.rotate(A, 90)
# print(A.x, A.y)
AB = r.points_to_line(A, B)
AC = r.points_to_line(B, C)

print(AB.a, AB.b, AB.c)
print(r.areIntersect(AB, AC)[0], r.areIntersect(AB, AC)[1].x, r.areIntersect(AB, AC)[1].y)

data_dir = 'classify/rectangle/label/'
files = os.listdir(data_dir)
for file in files[10000:]:
    img = cv2.imread(data_dir+file)
    res, res_seg = a.affineFinger(img, img) # the second parameter should be the segmentation
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(img)
    ax[1].imshow(res)
    ax[2].imshow(res_seg)
    plt.show()