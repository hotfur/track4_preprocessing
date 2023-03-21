import math
import random as rng
import os
import cv2 as cv
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue
from msrcr import retinex_FM
import shutil


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
    elif corners_check_shape == 7:
        shape_7.put((item_no[0], name[0]))
    elif corners_check_shape == 6:
        shape_6.put((item_no[0], name[0]))
    elif corners_check_shape == 5:
        shape_5.put((item_no[0], name[0]))
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
            result[past[0]] = [past[1]]
        elif past[0] == cur[0]:
            temp = result.get(past[0])
            temp.append(cur[1])
            result[past[0]] = temp
        else:
            past = cur
            result[past[0]] = [past[1]]
    return result


def writer(name, obj_type, white_check):
    path = '../dataset/train/'
    path_seg = '../dataset/segmentation_labels/'
    src_color = cv2.imread(cv2.samples.findFile(path + name + ".jpg"))
    src = cv2.imread(cv2.samples.findFile(path_seg + name + '_seg.jpg'))
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, src = cv2.threshold(src, 128, 1, cv2.THRESH_BINARY)
    cropt = cv2.bitwise_and(src_color, src_color, mask=src)
    src *= 255
    # Apply MSRCR
    cropt = cv2.GaussianBlur(cv2.bilateralFilter(retinex_FM(cropt, iter=4), d=9, sigmaSpace=50, sigmaColor=50), ksize=(3, 3), sigmaX=1)
    # Convert to Lab and calculate chromaticity distance
    # And check their distance to white objects
    mask = np.repeat(src[..., None], 3, axis=-1).astype(bool)
    img_Lab = cv2.cvtColor(cropt, cv2.COLOR_BGR2Lab)
    img_Lab[:, :, 0] = img_Lab[:, :, 0] * 0.9
    cropt = cv2.cvtColor(img_Lab, cv2.COLOR_Lab2BGR)
    var = np.var(img_Lab, axis=(0, 1), where=mask)
    mean = np.mean(img_Lab, axis=(0,1), where=mask)
    color_dist = np.sqrt(var[1]+var[2])
    white_dist = abs(mean[1] - 128) + abs(mean[2] - 128)

    # # Remove purely white objects
    # if np.mean(cropt, where=mask) > 255*0.8:
    #     cv2.imwrite("../dataset/classify/white_removal/label/" + name + ".jpg", src)
    #     cv2.imwrite("../dataset/classify/white_removal/seg/" + name + ".jpg", cropt)
    # If object has a white color, then it must be checked further to determine if
    # the 3d object is truly white object or just a noisy side of that object
    if (white_dist < 9.5):
        white_check.put(name.split("_")[0], [name, obj_type, white_dist])
        cv2.imwrite("../dataset/classify/white_check_pending/label/" + name + ".jpg", src)
        cv2.imwrite("../dataset/classify/white_check_pending/seg/" + name + ".jpg", cropt)
    # Remove monotonic surface objects
    elif (color_dist < 9.5):
        cv2.imwrite("../dataset/classify/color_removal/label/" + name + ".jpg", src)
        cv2.imwrite("../dataset/classify/color_removal/seg/" + name + ".jpg", cropt)
    # Write objects to their categories
    elif obj_type == "box":
        cv2.imwrite("../dataset/classify/box/label/" + name + ".jpg", src)
        cv2.imwrite("../dataset/classify/box/seg/" + name + ".jpg", cropt)
    elif obj_type == "abstract":
        cv2.imwrite("../dataset/classify/abstract/label/" + name + ".jpg", src)
        cv2.imwrite("../dataset/classify/abstract/seg/" + name + ".jpg", cropt)
    elif obj_type == "bottles":
        cv2.imwrite("../dataset/classify/bottles/label/" + name + ".jpg", src)
        cv2.imwrite("../dataset/classify/bottles/seg/" + name + ".jpg", cropt)


def white_object_writer(obj):
    path = "../dataset/classify/white_check_pending/label/"
    path_seg = "../dataset/classify/white_check_pending/seg/"

    obj_type = obj[0]
    # Write objects to their categories
    for name in obj[1]:
        if obj_type == "white_removal":
            shutil.copy(path + name + ".jpg", "../dataset/classify/white_removal/label/" + name + ".jpg")
            shutil.copy(path_seg + name + ".jpg", "../dataset/classify/white_removal/seg/" + name + ".jpg")
        elif obj_type == "box":
            shutil.copy(path + name + ".jpg", "../dataset/classify/box/label/" + name + ".jpg")
            shutil.copy(path_seg + name + ".jpg", "../dataset/classify/box/seg/" + name + ".jpg")
        elif obj_type == "abstract":
            shutil.copy(path + name + ".jpg", "../dataset/classify/abstract/label/" + name + ".jpg")
            shutil.copy(path_seg + name + ".jpg", "../dataset/classify/abstract/seg/" + name + ".jpg")
        elif obj_type == "bottles":
            shutil.copy(path + name + ".jpg", "../dataset/classify/bottles/label/" + name + ".jpg")
            shutil.copy(path_seg + name + ".jpg", "../dataset/classify/bottles/seg/" + name + ".jpg")

def highest_count_dict(k, dict_list):
    img_count = []
    value = []
    for dict in dict_list:
        img_list = dict.pop(k, None)
        if img_list is not None:
            value += img_list
            img_count.append(len(img_list))
        else:
            img_count.append(0)

    max_indx = np.argmax(img_count)
    return max_indx, value


if __name__ == "__main__":
    path_main = '../dataset/train/'
    files = os.listdir(path_main)
    shape_8 = queue.PriorityQueue()
    shape_7 = queue.PriorityQueue()
    shape_6 = queue.PriorityQueue()
    shape_5 = queue.PriorityQueue()
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
        dict_7 = executor.submit(decoder, shape_7).result()
        dict_6 = executor.submit(decoder, shape_6).result()
        dict_5 = executor.submit(decoder, shape_5).result()
        dict_4 = executor.submit(decoder, shape_4).result()
        dict_3 = executor.submit(decoder, shape_3).result()

    # Classification based on corner numbers and same objects
    boxes = {}
    bottles = {}
    abstract = {}
    dict_list = [dict_3, dict_4, dict_5, dict_6, dict_7, dict_8]
    for i in range(6):
        dict_keys = list(dict_list[i].keys())
        for k in dict_keys:
            max_indx, value = highest_count_dict(k, dict_list)
            if max_indx == 0:
                abstract[k] = value
            if max_indx == 1:
                boxes[k] = value
            if max_indx == 2:
                abstract[k] = value
            if max_indx == 3:
                boxes[k] = value
            if max_indx == 4:
                abstract[k] = value
            if max_indx == 5:
                bottles[k] = value

    # Result writer (white_object check pending)
    white_check = queue.PriorityQueue()
    with ThreadPoolExecutor() as executor:
        for v in boxes.values():
            for item in v:
                executor.submit(writer, item, "box", white_check)
        for v in abstract.values():
            for item in v:
                executor.submit(writer, item, "abstract", white_check)
        for v in bottles.values():
            for item in v:
                executor.submit(writer, item, "bottles", white_check)

    # Check for white objects
    pending_objs = decoder(white_check)
    processed_objs = queue.Queue()
    for obj in pending_objs.keys():
        num_img = len(pending_objs[obj])
        img_names_list = np.array(num_img)
        white_dist_list = np.array(num_img)
        type = pending_objs[obj][0][1]
        i = 0
        for img in pending_objs[obj]:
            img_names_list[i] = img[0]
            white_dist_list[i] = img[2]
            i+=1
        ptp = np.ptp(white_dist_list) # peek-to-peek value
        mean = np.mean(white_dist_list)
        # Definitely a color object
        if ptp > 9.5:
            processed_objs.put("white_removal", img_names_list[white_dist_list<9.5])
            processed_objs.put((type, img_names_list[white_dist_list>=9.5]))
        # Else it is a white object and we should only keep most valuable information
        # by comparing with the mean
        else:
            processed_objs.put("white_removal", img_names_list[white_dist_list < mean])
            processed_objs.put((type, img_names_list[white_dist_list >= mean]))

    # Remove white objects
    with ThreadPoolExecutor() as executor:
        while processed_objs.empty():
            executor.submit(white_object_writer, processed_objs.get())
