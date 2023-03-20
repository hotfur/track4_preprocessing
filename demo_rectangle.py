import rectangle as r
import randomFinger as rF
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
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

data_dir = 'segmentation_labels/'
files = os.listdir(data_dir)
for file in files[10000:]:
    img = cv2.imread(data_dir+file)
    res = rF.randomFinger(img, p = 1.)
    plt.imshow(res)
    plt.show()