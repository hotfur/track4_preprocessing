import math
import random as rng
import os
import cv2 as cv
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue



def findAngle(m1, m2):
    # Store the tan value  of the angle
    angle = abs((m2 - m1) / (1 + m1 * m2))
    # Calculate tan inverse of the angle
    ret = math.atan(angle)
    # Convert the angle from
    # radian to degree
    val = (ret * 180) / math.pi
    return round(val, 4)


def distancePoint(p1, p2):
    a = (p1[0] - p2[0]) * (p1[0] - p2[0])
    b = (p1[1] - p2[1]) * (p1[1] - p2[1])
    return math.sqrt(a + b)


def findSlope(p, q):
    if p[1] == q[1]:
        slope = 0
    elif p[0] == q[0]:
        slope = 1e9
    else:
        slope = (q[1] - p[1]) / (q[0] - p[0])
    intercept = p[1] - slope * p[0]
    return slope, intercept

def convert_hull(points):
    count = np.array(points, dtype=np.int32)
    arr = []
    for i in range(count.shape[0]):
        arr.append([count[i][0][0], count[i][0][1]])
    return arr

def convert_shape(points):
    count = np.array(points, dtype=np.int32)
    arr = []
    for i in range(count.shape[0]):
        arr.append([count[i][0][0], count[i][0][1]])
    return arr

def check_shape(src):
    gp_thres = 20
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(src, 100, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area >= 0.2 * src.shape[0] * src.shape[1]:
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 0.2 * src.shape[0] * src.shape[1]:
            # gives area of contour
            perimeter = cv2.arcLength(contour, closed=True)
            borders = cv2.approxPolyDP(curve=contour, epsilon=0.01 * perimeter, closed=False)
            borders = convert_hull(borders)

            temp = borders
            temp.append(borders[0])
            temp.append(borders[1])
            temp.append(borders[2])
            corners_check_shape = []
            arr_same_line = []
            for i in range(len(temp) - 1):

                if i + 2 < len(temp):
                    a = temp[i]
                    b = temp[i + 1]
                    c = temp[i + 2]

                    slope1, _ = findSlope(a, b)
                    slope2, _ = findSlope(b, c)
                    if abs((a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))) <= 0.02 * src.shape[
                        0] * src.shape[1]:
                        if len(arr_same_line) > 0:
                            a1 = arr_same_line[0]
                            b1 = temp[i + 1]
                            c1 = temp[i + 2]
                            if abs((a1[0] * (b1[1] - c1[1]) + b1[0] * (c1[1] - a1[1]) + c1[0] * (
                                    a1[1] - b1[1]))) > 0.02 * src.shape[0] * src.shape[1]:
                                corners_check_shape.append(arr_same_line[0])

                                arr_same_line = [temp[i + 1]]
                        else:
                            if temp[i] not in corners_check_shape:
                                arr_same_line.append(temp[i])
                    else:
                        if temp[i] not in arr_same_line:
                            arr_same_line.append(temp[i])
                        if len(arr_same_line) > 0:
                            if arr_same_line[0] not in corners_check_shape:
                                corners_check_shape.append(arr_same_line[0])
                            for k in range(1, len(arr_same_line)):
                                if arr_same_line[k] in corners_check_shape:
                                    corners_check_shape.remove(arr_same_line[k])
                            arr_same_line = []
                        else:
                            if temp[i] not in corners_check_shape:
                                corners_check_shape.append(temp[i])
                else:
                    if len(arr_same_line) > 0:
                        if arr_same_line[0] not in corners_check_shape:
                            corners_check_shape.append(arr_same_line[0])
                        arr_same_line = []
                    else:
                        if temp[i] not in corners_check_shape:
                            corners_check_shape.append(temp[i])
            temp = corners_check_shape
            corners_check_shape = []
            dict_label = {}
            flag = [0] * len(temp)
            for i in range(len(temp)):
                for j in range(i + 1, len(temp)):
                    dis = distancePoint(temp[i], temp[j])
                    if dis <= gp_thres:
                        flag[i] = i + 1
                        flag[j] = i + 1
            for i in range(len(temp)):
                if flag[i] == 0:

                    corners_check_shape.append(temp[i])
                elif flag[i] < 1e9:
                    cnt = 1
                    p = temp[i]

                    for j in range(i + 1, len(temp)):
                        if flag[i] == flag[j]:
                            p[0] += temp[j][0]
                            p[1] += temp[j][1]
                            cnt += 1
                            flag[j] = 1e9
                    p[0] = int(p[0] / cnt)
                    p[1] = int(p[1] / cnt)
                    flag[i] = 1e9
                    corners_check_shape.append(p)
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            for corner in corners_check_shape:
                cv.circle(drawing, (corner[0], corner[1]), 3, color, -1)
            return len(corners_check_shape), corners_check_shape
    return None


def mainfunc(f):
    """Old main function"""
    path = '../dataset/train/'
    path_seg = '../dataset/segmentation_labels/'
    name = f.split('.')
    src_color = cv2.imread(cv2.samples.findFile(path + f))
    default_file = path_seg + name[0] + '_seg.' + name[1]
    src = cv2.imread(cv2.samples.findFile(default_file))
    corners_check_shape, corners = check_shape(src)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, src = cv2.threshold(src, 128, 1, cv2.THRESH_BINARY)
    cropt = cv2.bitwise_and(src_color, src_color, mask=src)
    src *= 255
    if corners_check_shape > 7:
        cv2.imwrite('../dataset/classify/other/label/' + name[0] + '_mask.jpg', src)
        cv2.imwrite('../dataset/classify/other/seg/' + name[0] + '_seg.jpg', cropt)
    elif corners_check_shape == 6:
        cv2.imwrite("../dataset/classify/rectangle_like/label/" + name[0] + "_mask.jpg", src)
        cv2.imwrite("../dataset/classify/rectangle_like/seg/" + name[0] + "_mask.jpg", cropt)
    elif corners_check_shape == 4:
        cv2.imwrite("../dataset/classify/rectangle/label/" + name[0] + "_mask.jpg", src)
        cv2.imwrite("../dataset/classify/rectangle/seg/" + name[0] + "_mask.jpg", cropt)
    else:
        cv2.imwrite("../dataset/classify/unknown/label/" + name[0] + "_mask.jpg", src)
        cv2.imwrite("../dataset/classify/unknown/seg/" + name[0] + "_mask.jpg", cropt)


def encoder(f):
    """Encode the dataset into queue of items, each queue contains only items of specific corner counts.
     Each item is the name of each image in the dataset"""
    path_seg = '../dataset/segmentation_labels/'
    name = f.split('.')
    default_file = path_seg + name[0] + '_seg.' + name[1]
    src = cv2.imread(cv2.samples.findFile(default_file))
    corners_check_shape, corners = check_shape(src)
    # Encoder: pack shape & item_no together to classify item type
    item_no = f.split("_")
    if corners_check_shape > 7:
        shape_8.put((item_no[0], name[0]))
    elif (corners_check_shape == 7) or (corners_check_shape == 6) or (corners_check_shape == 5):
        shape_6.put((item_no[0], name[0]))
    elif corners_check_shape == 4:
        shape_4.put((item_no[0], name[0]))
    else:
        shape_3.put((item_no[0], name[0]))

def decoder(q):
    """Decode a queue with duplicated items into a dict containing only counts of such items"""
    result = {}
    past = None
    while not q.empty():
        cur = q.get()
        if past is None:
            past = cur
            past[past[0]] = [past[1]]
        elif past[0] == cur[0]:
            temp = result.get(past[1])
            temp.append(cur[1])
            result[past[0]] = temp
        else:
            past = cur
            past[past[0]] = [past[1]]
    return result


def writer(name, obj_type):
    path = '../dataset/train/'
    path_seg = '../dataset/segmentation_labels/'
    src_color = cv2.imread(cv2.samples.findFile(path + name + ".jpg"))
    src = cv2.imread(cv2.samples.findFile(path_seg + name + '_seg.jpg'))
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, src = cv2.threshold(src, 128, 1, cv2.THRESH_BINARY)
    cropt = cv2.bitwise_and(src_color, src_color, mask=src)
    src *= 255
    if obj_type == "box":
        cv2.imwrite("../dataset/classify/box/label/" + name[0] + "_mask.jpg", src)
        cv2.imwrite("../dataset/classify/box/seg/" + name[0] + "_mask.jpg", cropt)
    elif obj_type == "abstract":
        cv2.imwrite("../dataset/classify/abstract/label/" + name[0] + "_mask.jpg", src)
        cv2.imwrite("../dataset/classify/abstract/seg/" + name[0] + "_mask.jpg", cropt)
    else:
        cv2.imwrite("../dataset/classify/unknown/label/" + name[0] + "_mask.jpg", src)
        cv2.imwrite("../dataset/classify/unknown/seg/" + name[0] + "_mask.jpg", cropt)


if __name__ == "__main__":
    path_main = '../dataset/train/'
    files = os.listdir(path_main)
    shape_8 = queue.PriorityQueue()
    shape_6 = queue.PriorityQueue()
    shape_4 = queue.PriorityQueue()
    shape_3 = queue.PriorityQueue()
    # for f in range(10):
    #     mainfunc(files[f])

    # Encode
    with ThreadPoolExecutor() as executor:
        for f in files:
            executor.submit(encoder, f)
    # Decode
    with ThreadPoolExecutor() as executor:
        dict_8 = executor.submit(decoder, shape_8).result()
        dict_6 = executor.submit(decoder, shape_6).result()
        dict_4 = executor.submit(decoder, shape_4).result()
        dict_3 = executor.submit(decoder, shape_3).result()

    # Classification based on corner numbers and same objects
    boxes = {}
    abstract = {}
    unknown = {}
    for k in dict_6.keys():
        if k in dict_4:
            boxes[k] = dict_4.pop(k) + dict_6.pop(k)
            if k in dict_3:
                boxes[k] += dict_3.pop(k)
            if k in dict_8:
                boxes[k] += dict_8.pop(k)
        elif k in dict_3:
            unknown[k] = dict_6.pop(k) + dict_3.pop(k)
            if k in dict_8:
                abstract[k] = dict_8.pop(k) + unknown.pop(k)
        elif k in dict_8:
            unknown[k] = dict_6.pop(k) + dict_8.pop(k)
        else:
            boxes[k] = dict_6[k]
    for k in dict_3.keys():
        if k in dict_4:
            boxes[k] = dict_3.pop(k) + dict_4.pop(k)
        if k in dict_8:
            abstract[k] = dict_3.pop(k) + dict_8.pop(k)
        else:
            unknown[k] = dict_3.pop(k)
    for k in dict_4.keys():
        if k in dict_3:
            boxes[k] = dict_3.pop(k) + dict_4.pop(k)
        if k in dict_8:
            abstract[k] = dict_4.pop(k) + dict_8.pop(k)
        else:
            boxes[k] = dict_4.pop(k)
    for k in dict_8.keys():
        abstract[k] = dict_8.pop(k)

    # Result writer
    with ThreadPoolExecutor() as executor:
        for v in boxes.values():
            executor.submit(writer, v, "box")
        for v in abstract.values():
            executor.submit(writer, v, "abstract")
        for v in unknown.values():
            executor.submit(writer, v, "unknown")
