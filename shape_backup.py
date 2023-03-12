"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import cv2 as cv
import cv2
import numpy as np
import math

import random as rng
# from count_line import findAngle 
def findAngle(M1, M2):
    PI = 3.14159265
     
    # Store the tan value  of the angle
    angle = abs((M2 - M1) / (1 + M1 * M2))
 
    # Calculate tan inverse of the angle
    ret = math.atan(angle)
 
    # Convert the angle from
    # radian to degree
    val = (ret * 180) / PI
 
    # Print the result
    return (round(val, 4))

def find_line(a, b, c, d):
    return [b-d, c - a, -a*(b-d) - b *(c-a)]

def check_line(line, p1, p2):
    value1 = line[0]*p1[0] + line[1]*p1[1] + line[2]
    value2 = line[0]*p2[0] + line[1]*p2[1] + line[2]
    return [value1, value2]

def check_line_P(line, p1):
    # print(line, p1)
    value1 = abs(line[0]*p1[0] + line[1]*p1[1] + line[2])
    result = value1/(math.sqrt(line[0]*line[0] + line[1]*line[1]))
    return result

def gr_line_parallels(gr_lines, pts, thresh):
    rs_gr = []
    rs_pts = []
    for i in range(len(gr_lines)):
        if len(rs_gr) == 0:
            rs_gr.append(gr_lines[i])
            rs_pts.append([pts[i][0], pts[i][1] ,pts[i][2], pts[i][3]])
        else:
            check = True
            # for rs in rs_gr:
            for j in range(len(rs_gr)):
                a1 = abs(rs_gr[j][0] - gr_lines[i][0])
                a2 = abs(rs_gr[j][1] - gr_lines[i][1])
                a3 = abs(rs_gr[j][2] - gr_lines[i][2])
                # dis1 = sqrt( (pts[i][0]-rs_pts[i][0])*(pts[i][0]-rs_pts[i][0]) + (pts[i][1]-rs_pts[i][1])*(pts[i][1]-rs_pts[i][1]))
                # dis2 = sqrt( (pts[i][2]-rs_pts[i][2])*(pts[i][2]-rs_pts[i][2]) + (pts[i][3]-rs_pts[i][3])*(pts[i][3]-rs_pts[i][3]))
                if a1 <= thresh and a2 <=thresh and a3 <= 2000:
                    check = False
                    break
            if check == True:
                rs_gr.append(gr_lines[i])
                rs_pts.append([pts[i][0], pts[i][1] ,pts[i][2], pts[i][3]])
    return rs_gr, rs_pts
def findAngle(M1, M2):
    PI = 3.14159265
     
    # Store the tan value  of the angle
    angle = abs((M2 - M1) / (1 + M1 * M2))
 
    # Calculate tan inverse of the angle
    ret = math.atan(angle)
 
    # Convert the angle from
    # radian to degree
    val = (ret * 180) / PI
 
    # Print the result
    return (round(val, 4))

def check_line_extra(line1, line2, thresh):
    # print(line1, line2)
    a1 = line1[0]
    b1 = line1[1]
    a2 = line2[0]
    b2 = line2[1]

    

    if b1 != 0 and b2 != 0:
        value1 = a1/b1
        value2 = a2/b2
        slop1 = -a1/b1
        slop2 = -a2/b2
        angle = findAngle(slop1, slop2)
        a1 = abs(line1[0] - line2[0])
        a2 = abs(line1[1] - line2[1])
        a3 = abs(line1[2] - line2[2])
        # dis1 = sqrt( (pts[i][0]-rs_pts[i][0])*(pts[i][0]-rs_pts[i][0]) + (pts[i][1]-rs_pts[i][1])*(pts[i][1]-rs_pts[i][1]))
        # dis2 = sqrt( (pts[i][2]-rs_pts[i][2])*(pts[i][2]-rs_pts[i][2]) + (pts[i][3]-rs_pts[i][3])*(pts[i][3]-rs_pts[i][3]))
        # if a1 <= 100 and a2 <=100 and a3 <= 2000:
        if angle <= 10:

            return True
            
            # break
            # else:
            #     return False
        else:
            return False
    else:
        if(abs(a1-a2) <= 5 and abs(b1-b2) <= 5):
              #If it is true then return True else return False.
            return True
        else:
            return False


def check_line_parralell(line1, line2):
    a1 = line1[0]
    b1 = line1[1]
    a2 = line2[0]
    b2 = line2[1]
    if b1 != 0 and b2 != 0:
        value1 = a1/b1
        value2 = a2/b2
        slop1 = -a1/b1
        slop2 = -a2/b2
        angle = findAngle(slop1, slop2)
        print(angle)
        if angle <= 10:
            return True
        else:
            return False
    else:
        if(abs(a1-a2) <= 5 and abs(b1-b2) <= 5):
              #If it is true then return True else return False.
            return True
        else:
            return False

def check_line_parralell2(line1, line2):
    a1 = line1[0]
    b1 = line1[1]
    a2 = line2[0]
    b2 = line2[1]
    if b1 != 0 and b2 != 0:
        value1 = a1/b1
        value2 = a2/b2
        slop1 = -a1/b1
        slop2 = -a2/b2
        angle = findAngle(slop1, slop2)
        print(angle)
        if angle <= 10:
            return True
        else:
            return False
    else:
        if(abs(a1-a2) <= 5 and abs(b1-b2) <= 5):
              #If it is true then return True else return False.
            return True
        else:
            return False

def distancePoint(p1, p2):
    a = (p1[0]-p2[0])*(p1[0]-p2[0])
    b = (p1[1]-p2[1])*(p1[1]-p2[1])
    # print(a+b)
    return math.sqrt(a+b)


def edge_detection(min_len, value_hough, f, file, file_color, min_gap, hough_phi):
    image = extrat_mask(f, file, file_color)
    # cv.imshow("image", image)
    kernel = np.array([
                    [0, -1, 0],
                    [-1, 5,-1],
                    [0, -1, 0]])
    # image = cv.filter2D(src=image, ddepth=-1, kernel=kernel)
    src = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    src = cv.GaussianBlur(src,(3,3),0)
    edges = cv.Canny(src,100,255)
    # cv.imshow("edges", edges)
    cdstP = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    linesP = cv.HoughLinesP(edges, 1, np.pi / value_hough, hough_phi, None, min_len, min_gap) 
    # print(len(linesP))
    if linesP is not None:

        # print(len(linesP))
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # print(i, l)
            temp = cdstP.copy()
            color = list(np.random.random(size=3) * 256)


            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), color, 3, cv.LINE_AA)
    cv.imshow("cdstP cvtColor", cdstP)       
    return linesP     

def find_hull(points):
    count = np.array(points, dtype=np.int32)           
    hull = cv.convexHull(count)
    arr = []
    for i in range(hull.shape[0]):
        # if int(point_start[0]) == hull[i][0][0] and int(point_start[1]) == hull[i][0][1]:
        #     # index_start = i
        #     point_start = [hull[i][0][0], hull[i][0][1]]
        arr.append([hull[i][0][0], hull[i][0][1]])
    return arr

def findPoints(source, m, l):
    a = [0, 0]
    b = [0, 0]

    if m == 0:
        a[0] = source[0] + l
        a[1] = source[1]

        b[0] = source[0] - l
        b[1] = source[1]

    elif m == 1e9:
        a[0] = source[0]
        a[1] = source[1] + l
        b[0] = source[0]
        b[1] = source[1] - l
    else:
        dx = (l/math.sqrt(1+(m*m)))
        dy = m*dx
        a[0] = source[0] + dx
        a[1] = source[1] + dy
        b[0] = source[0] - dx
        b[1] = source[1] - dy
    a[0] = math.ceil(a[0])
    a[1] = math.ceil(a[1])
    
    b[0] = math.ceil(b[0])
    b[1] = math.ceil(b[1])
    return a, b

def findSlope(p, q):
    slope = 0
    if p[1] == q[1]:
        slope = 0
    elif p[0] == q[0]:
        slope = 1e9
    else:
        slope = (q[1]-p[1])/(q[0]-p[0])
    intercept = p[1] - slope*p[0]
    return slope, intercept

    # return (q[1]-p[1])/(q[0]-p[0])

def findMissingPoint(a, b, c):
    print(a,b,c)
    slope, _ = findSlope(b,c)
    p1, p2 = findPoints(a, slope, distancePoint(b,c))
    p = []
    if distancePoint(p1, c) - distancePoint(a,b) <= 10:
        p = [p1]
    if distancePoint(p2, c) - distancePoint(a,b) <= 10:
        p = [p1, p2]
    return p

def findNearestWhite(image, points):
    mine = 1e9
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if image[j,i] > 0:
                mine = min(mine, distancePoint([i, -j], points))

    return mine

def findNearestWhiteRemove(image, points):
    mine = 1e9
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if image[j,i] > 0:
                mine = min(mine, distancePoint([i, j], points))

    return mine
def angle(a, b, c):
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(b)
    print(a,b,c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)    
    # print(np.degrees(angle))
    return angle

# def check_shape(gppoints):
#     hull = find_hull(gppoints)
#     print(hull)
#     an1 = angle(hull[0], hull[1], hull[2])
#     print(an1)
#     # an2 = angle(hull[1], hull[2], hull[3])
#     # an3 = angle(hull[2], hull[3], hull[0])
#     # an4 = angle(hull[3], hull[0], hull[1])
#     # print(an1, an2, an3, an4)
#     # return max(an1, an2, an3,an4)

def extrat_mask(f, file, file_color):
    name = f.split('.')
    src_color = cv.imread(cv.samples.findFile(file_color+f))
    default_file = file +name[0] +'_seg.' + name[1]
    src = cv.imread(cv.samples.findFile(default_file))
    
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src[src>0]=255
    cropt = cv.bitwise_and(src_color, src_color, mask=src)
    # cv.imshow("mask cropt "+ str(i), cropt)
    # cv.imwrite('D:/IU_Studying/research/New folder (2)/Auto-retail-syndata-release/other/'+name[0]+'.jpg', cropt)
    return cropt

def check_nearCorner(point, corners):
    mine = 1e9
    for corner in corners:
        dis = distancePoint(point, corner)
        if dis != 0:
            # print(point, corner, dis)
            mine = min(dis, mine)
    return mine

# def find_total_line(lines, points):
#     rs_lines = []
#     rs_points = []
#     for i in range(len(lines)):
#         if len(rs_lines) == 0:
#             rs_lines.append(lines[i])
#             rs_points.append(points[i])
#         else:
#             for j in range(len(rs_lines)):
#                 if check_line_parralell(lines[i], rs_lines[j]):
#                     ps = points[i]
#                     dis = min(check_line_P( rs_lines[j], (ps[0][0], -ps[0][1])), check_line_P(rs_lines[j], (ps[1][0], -ps[1][1])))
#                     if dis > 100:
#                         rs_lines.append(lines[i])
#                         rs_points.append(points[i])
#                     else:
#                         if distancePoint((ps[0][0], -ps[0][1]), (ps[1][0], -ps[1][1])) > distancePoint((rs_points[i][0][0], -rs_points[i][0][1]), (rs_points[i][1][0], -rs_points[i][1][1])):
#                             rs_lines[j] = lines[i]
#                             rs_points[j] = points[i]
#     return rs_lines, rs_points

                        
                     
def convert_hull(points):
    count = np.array(points, dtype=np.int32)           
    hull = cv.convexHull(count)
    arr = []
    for i in range(count.shape[0]):
        # if int(point_start[0]) == hull[i][0][0] and int(point_start[1]) == hull[i][0][1]:
        #     # index_start = i
        #     point_start = [hull[i][0][0], hull[i][0][1]]
        arr.append([count[i][0][0], count[i][0][1]])
    return arr

def convert_shape(points):
    count = np.array(points, dtype=np.int32)           
    # hull = cv.convexHull(count)
    arr = []
    for i in range(count.shape[0]):
        # if int(point_start[0]) == hull[i][0][0] and int(point_start[1]) == hull[i][0][1]:
        #     # index_start = i
        #     point_start = [hull[i][0][0], hull[i][0][1]]
        arr.append([count[i][0][0], count[i][0][1]])
    return arr

def check_shape(src):
    gp_thres = 20
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(src, 100, 255, 0)
    # thresh = cv.Canny(thresh, 50, 255, None, 3)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE )
    drawing = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area >= 0.2*src.shape[0]*src.shape[1]:
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            perimeter = cv.arcLength(contours[i],True)
            # print(perimeter)
            cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    # print(len(contours))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 0.2*src.shape[0]*src.shape[1]:
            # gives area of contour
            perimeter = cv2.arcLength(contour, closed=True)
            # gives perimeter of contour
            # print(contour)
            borders = cv2.approxPolyDP(curve=contour,epsilon=0.01*perimeter,closed=False)
            borders = convert_hull(borders)
            # print(len(borders))
            # print(borders)
            # print(area)
            contour_temp = convert_shape(contour)
            

            temp = borders
            temp.append(borders[0])
            temp.append(borders[1])
            temp.append(borders[2])
            corners_check_shape = []
            # print("checking same line", len(temp))
            arr_same_line = []
            for i in range(len(temp)-1):
                

                if i + 2 < len(temp):
                    a = temp[i]
                    b = temp[i+1]
                    c = temp[i+2]
                    
                    slope1,_ = findSlope(a, b)
                    slope2,_ = findSlope(b,c)
                    angle = findAngle(slope1, slope2)
                    # print(abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))), a, b, c, angle, 0.02*src.shape[0]*src.shape[1])
                    if abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))) <=  0.02*src.shape[0]*src.shape[1]:
                        print(abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))), a, b, c)

                    # if abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))) <=  500:
                        # print(i)
                        if len(arr_same_line) > 0:
                            a1 = arr_same_line[0]
                            b1 = temp[i+1]
                            c1 = temp[i+2]
                            if abs((a1[0]*(b1[1]-c1[1]) + b1[0]*(c1[1]-a1[1]) + c1[0]*(a1[1]-b1[1]))) >  0.02*src.shape[0]*src.shape[1]:
                                corners_check_shape.append(arr_same_line[0])
                                
                                arr_same_line = [temp[i+1]]
                                # print("check Huong ", arr_same_line, corners_check_shape)
                        else:
                            if temp[i] not in corners_check_shape:
                                arr_same_line.append(temp[i])
                        # print(arr_same_line, temp[i])
                    else:
                        if temp[i] not in arr_same_line:
                            arr_same_line.append(temp[i])
                        # print(len(arr_same_line), arr_same_line)
                        if len(arr_same_line) > 0:
                            # print(len(arr_same_line), arr_same_line)
                            if arr_same_line[0] not in corners_check_shape:
                                corners_check_shape.append(arr_same_line[0])
                            for k in range(1, len(arr_same_line)):
                                if arr_same_line[k] in corners_check_shape:
                                    corners_check_shape.remove(arr_same_line[k])
                            arr_same_line = []
                        else:
                            
                            # print(i)
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
                # print(arr_same_line)
            # print(len(corners))
            temp = corners_check_shape
            corners_check_shape = []
            dict_label = {}
            flag = [0]*len(temp)
            # for i in range(len(temp)):
            #     dict_label[tuple(temp[i])] = 0
            for i in range(len(temp)):

                # print(src[int(temp[i][1]), int(abs(temp[i][0]))])
                for j in range(i+1, len(temp)):
                    dis = distancePoint(temp[i], temp[j])
                    # print(dis)
                    if dis <= gp_thres:
                        # if tuple(temp[i]) in dict_label.keys():
                            
                        #     flag[j] = dict_label[tuple(temp[i]) ]
                        # else:
                        #     dict_label[tuple(temp[i])] = i+1
                        #     dict_label[tuple(temp[j])] = i+1
                            # flag[i] = i+1
                        # if j == 12:
                        #     print(dis, i)
                        flag[i] = i+1
                        flag[j] = i+1

            # print(flag)
            for i in range(len(temp)):
                # print(i , flag[i])
                if flag[i] == 0:
                    
                    corners_check_shape.append(temp[i])
                    # print(i)
                elif flag[i] < 1e9:
                    cnt = 1
                    p = temp[i]
                    # print(i)
                    # print(p)
                    
                    for j in range(i + 1, len(temp)):
                        if flag[i] == flag[j]:
                        # if dict_label[tuple(temp[i])] == dict_label[tuple(temp[j])]:
                            print(i, j)
                            p[0] += temp[j][0]
                            p[1] += temp[j][1]
                            cnt += 1
                            flag[j] = 1e9
                    # print(p)
                    p[0] = int(p[0]/cnt)
                    p[1] = int(p[1]/cnt)
                    flag[i] = 1e9
                    # print(flag)
                    corners_check_shape.append(p)
            # print(len(corners_check_shape))
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            for corner in corners_check_shape:
                # pass
                cv.circle(drawing,(corner[0], corner[1]), 3, color, -1)
            if len(corners_check_shape) > 7:
                print("Other")
            else:
                print("Box shape")
            # cv.imshow('Contours2' +f, drawing2)
            # cv.imshow('Contours' +f, drawing)
            # cv.imshow("thresh"+ f, thresh)
            return len(corners_check_shape), corners_check_shape
    return None


def select_best_inside(candidates, line_cnt, corners_temp):
    corners = corners_temp.copy()
    for candidate in candidates:
        line_chosen = line_cnt[(candidate[0][0], candidate[0][1])]
        # print(line_chosen)
        point_start = line_chosen[0]
        # print(point_start)
        # value = check_line_P(line_chosen, (corners[0][0], -corners[0][1]))
        index_start = 0 
        # for i in range(1, len(corners)):
        #     value_temp = check_line_P(line_chosen, (corners[i][0], -corners[i][1]))
        #     print(value_temp, value, corners[i])
        #     if value_temp < value:
        #         value = value_temp
        #         point_start = corners[i]
        #         index_start = i
                
        

        # corners.append([list(sort_orders[0])[0][0], -list(sort_orders[0])[0][1]])
        
        final_corner = []
        for corner in corners:
            final_corner.append( [int(corner[0]), abs(int(corner[1]))])
        # print(len(final_corner))
        print("------------------")
        print(candidate)
        print(point_start)
        # print(final_corner)
        inside = [int(list(candidate)[0][0]), abs(int(list(candidate)[0][1]))]
        # print(inside)
        count = []
        
        count = np.array(final_corner, dtype=np.int32)           
        hull = cv.convexHull(count)
        # print(hull.shape)
        # print(hull[0][0][0], hull[0][0][1])
        # print(hull)
        arr = []
        for i in range(hull.shape[0]):
            if int(point_start[0]) == hull[i][0][0] and int(point_start[1]) == -hull[i][0][1]:
                index_start = i
                point_start = [hull[i][0][0], hull[i][0][1]]
            arr.append([hull[i][0][0], hull[i][0][1]])
        
        # print(point_start, index_start)
        index = index_start + 1
        slice_corner = [point_start]
        if index == len(arr):
            index = 0
        while index != index_start:
            # print(index)
            
            slice_corner.append(arr[index])
            index = index + 1
            if index == len(arr):
                index = 0
        # print(corners)
        slice_corner.append(point_start)
        # print(slice_corner)
        gr_corners = [slice_corner[0]]
        box_shapes = []
        index = 1
        while index < len(slice_corner):
        # for i in range(1, len(slice_corner)):
            # print(index)
            if len(gr_corners) == 2:
                # print("checking shape")
                gr_corners.append(slice_corner[index])
                gr_corners.append(inside)
                # print(gr_corners)
                hulls = find_hull(gr_corners)
                if len(hulls) == 4:
                    box_shapes.append(gr_corners)
                    # print(gr_corners)
                    # print(check_shape(gr_corners))
                    gr_corners = [slice_corner[index]]
                    index = index + 1
                else:
                    
                    index = index - 1
                    # print(i)
                    gr_corners= [slice_corner[index]]
                    index = index + 1
                    # print(gr_corners)
            else:
                gr_corners.append(slice_corner[index])
                index = index + 1


        # print(box_shapes)
        contours = []
        check = True
        for box in box_shapes:
            # print(box)
            a = box[0]
            b = box[1]
            c = box[2]
            d = box[3]

            line1 = find_line(a[0], -a[1], b[0], -b[1])
            line2 = find_line(d[0], -d[1], c[0], -c[1])

            
            line3 = find_line(b[0], -b[1], c[0], -c[1])
            line4 = find_line(a[0], -a[1], d[0], -d[1])
            print(distancePoint(a, b), distancePoint(d, c), distancePoint(b, c), distancePoint(a, d))
            check1 =  check_line_parralell2(line1, line2)
            check2 =  check_line_parralell2(line3, line4)
            check_temp = True
            if not check1 or not check2:
                check_temp = False
            check = check and check_temp

        if check == True:
            return candidate
    print("final check", check)
    return candidates[0]

def main(argv, f, file, file_color, corners_shape):
    thresh = 200
    gp_thres = 20
    value_hough = 60
    min_gap = 40
    hough_phi = 50
    name = f.split('.')
    # src_color = extrat_mask(f, file, file_color)
    src_color = cv.imread(cv.samples.findFile(file_color+f))
    default_file = file +name[0] +'_seg.' + name[1]
    src = cv.imread(cv.samples.findFile(default_file))
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    min_len = min(src.shape[0], src.shape[1])/4
    # src_color = cv.imread(cv.samples.findFile('Auto-retail-syndata-release/syn_image_train/'+file))
    
    
    
    kernel = np.ones((5,5),np.uint8)
    # src = cv.erode(src,kernel,iterations = 1)
    if src.shape[0]*src.shape[1] <= 150*150:
        value_hough = 10
        thresh = 20
        gp_thres = 10
        min_gap = 20
        hough_phi = 20
    linesP_color = edge_detection(min_len, value_hough, f, file, file_color, min_gap, hough_phi)
    kernelerode = np.ones((5,5),np.uint8)
    # src = cv.erode(src,kernelerode,iterations = 1)
    ori = src.copy()
    ori2 = src.copy()
    ret,ori2 = cv.threshold(ori2,1,255,cv.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    # ori = cv.erode(ori,kernel,iterations =3)
    ori2 = cv.dilate(ori2,kernel,iterations =1)
    # Check if image is loaded fine1
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1


    #result is dilated for marking the corners, not important
    kernel1 = np.ones((3,3),np.uint8)
    
    # Threshold for an optimal value, it may vary depending on the image.

    
    dst = cv.Canny(src, 100, 200, None, 3)
    # dst = cv.dilate(dst,kernel1,iterations = 1)
    # cv.imshow('dst',dst)
    
    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    # print(src.shape)
    
    gr_lines = []
    line_no = []
    pts = []
    linesP = cv.HoughLinesP(dst, 2, np.pi / value_hough, 10, None, 10, 40) 
    # print(len(linesP))
    temp_label_edges = cdstP.copy()
    if linesP is not None:

        # print(len(linesP))
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # print(i, linesP[i])
            temp = cdstP.copy()
            color = list(np.random.random(size=3) * 256)


            # cv.line(temp_label_edges, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            cv.line(temp, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            # cv.imwrite("test_line/" + str(i) + ".jpg", temp)
            
            # print(i, find_line(l[0], -l[1], l[2], -l[3]))
            # cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (10, 30, 255), 3, cv.LINE_AA)
            # cv.circle(cdstP,(l[0], l[1]), 1, (0,0,255), -1)
            # cv.circle(cdstP,(l[2], l[3]), 1, (0,0,255), -1)
            # print(find_line(l[0], l[1], l[2], l[3]))
            if len(gr_lines) == 0:
                gr_lines.append(find_line(l[0], -l[1], l[2], -l[3]))
                line_no.append(i)
                # print(find_line(l[0], -l[1], l[2], -l[3]))
                cv.line(temp_label_edges, (l[0], l[1]), (l[2], l[3]), (10, 30, 255), 3, cv.LINE_AA)
                # cv.circle(cdstP,(l[0], l[1]), 1, (0,0,255), -1)
                # cv.circle(cdstP,(l[2], l[3]), 1, (0,0,255), -1)
                pts.append(l)
            else:
                # print(len(gr_lines))
                temp = None
                check = True
                for gr_line in gr_lines:
                    v1, v2 = check_line(gr_line, (l[0], -l[1]), (l[2], -l[3]))
                    temp_line = find_line(l[0], -l[1], l[2], -l[3])
                    # print(v1, v2)
                    v1 = abs(v1)
                    v2 = abs(v2)
                    if v1 <= 500 and v2 <= 500 :
                        # cv.line(temp_label_edges, (l[0], l[1]), (l[2], l[3]), (10, 30, 255), 3, cv.LINE_AA)
                        # print(gr_line, temp_line)
                        check = False
                        break

                        # break
                if check == True:    
                    gr_lines.append(find_line(l[0], -l[1], l[2], -l[3]))
                    # print(find_line(l[0], -l[1], l[2], -l[3]))
                    # cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (10, 30, 255), 3, cv.LINE_AA)
                    line_no.append(i)
                    pts.append(l)
                        

    # cv2.imshow("temp_label_edges", temp_label_edges)
    # # cv2.waitKey(0)
    # # print(len(gr_lines))
    # for i in range(len(gr_lines)):
    #     print(i, gr_lines[i], pts[i][0], pts[i][1], pts[i][2], pts[i][3])
    # # for i in range(len(gr_lines)):
    #     # temp = ori.copy()
    #     # color = list(np.random.random(size=3) * 256)


    #     # cv.line(temp, (pts[i][0], pts[i][1]), (pts[i][2], pts[i][3]), (100,100,100), 3, cv.LINE_AA)
    #     # cv.imwrite("test_line/gr_ln" + str(i) + ".jpg", temp)
    # rs_line, rs_pts = gr_line_parallels(gr_lines, pts, thresh)
    # ex_tra_line = []
    # ex_tra_pt = []
    temp_find = ori.copy()
    temp_find = cv.cvtColor(temp_find, cv.COLOR_GRAY2BGR)
    

    # # for i in range(len(ex_tra_line)):
    # #     gr_lines.append(ex_tra_line[i])
    # #     pts.append(ex_tra_pt[i])
    # rs_line, rs_pts = gr_lines, pts
    # print(len(rs_line))
    # for i in range(len(rs_line)):
    #     print(rs_line[i], rs_pts[i][0], rs_pts[i][1], rs_pts[i][2], rs_pts[i][3])
    # # for no in line_no:
    # #     print(no)
    # for i in range(len(rs_line)):
    #     temp = src_color.copy()
    #     color = list(np.random.random(size=3) * 256)
    #     cv.line(temp, (rs_pts[i][0], rs_pts[i][1]), (rs_pts[i][2], rs_pts[i][3]), (55,100,255), 3, cv.LINE_AA)
    #     cv.imwrite("test_line/gr_ln" + str(i) + ".jpg", temp)
    # cv.imwrite("test.jpg", cdstP)
    # point_corners = []
    # corners = []
    # for i in range(len(rs_line)):
    #     for j in range(i+1, len(rs_line)):
    #         a1 = abs(rs_line[j][0] - rs_line[i][0])
    #         a2 = abs(rs_line[j][1] - rs_line[i][1])
    #         print(i, j)
    #         # if a1 == 0 or a2 == 0:
    #         try:
    #             A = np.array([[rs_line[i][0], rs_line[i][1]], [rs_line[j][0], rs_line[j][1]]])
    #             B = np.array([-rs_line[i][2],-rs_line[j][2]])
    #             solve = np.linalg.solve(A,B).tolist()
      
    #             if abs(solve[0]) - src.shape[1] <= 30 and abs(solve[1]) - src.shape[0] <= 30:
                    
    #                 if solve[0] >= 0 and solve[0] <= src.shape[1] and solve[1] <= 0 and abs(solve[1]) <= src.shape[0]:
    #                     corners.append(solve)
    #                 else:
                        
    #                     point_corners.append(solve)
    #         except:
    #             print("no solution")
    # print(len(point_corners))
    # for corner in point_corners:
    #     print(corner)
    # print(len(corners))
    
    # for corner in corners:
    #     print(corner)


    # temp_corners = []

    # for point in point_corners:
    #     cnt = 0
    #     minDis = 1e9
    #     for corner in corners:
    #         # print(point, corner, distancePoint(point, corner))
    #         distance = distancePoint(point, corner)
    #         minDis = min(distance, minDis)
    #         # if distance <= 30:
    #         #     cnt = cnt + 1
    #     # print('minDis',minDis)
    #     if cnt < 2:
    #         # print(cnt, point)
    #         check = False
    #         if point[0] < 0 and point[1] < 0 and abs(point[0]) <= 30 and abs(min(point[1], -src.shape[0])) - src.shape[0]  <= 30:
    #             point[0] = 0
    #             point[1] = max(point[1], -src.shape[0]+1)
    #             check = True
    #         elif point[0] > 0 and point[1] > 0 and point[1] <= 30 and max(point[0], src.shape[1]) - src.shape[1] <= 30:
    #             point[1] = 0
    #             point[0] = min(point[0], src.shape[1]-1)
    #             check = True
    #         elif point[0] < 0 and point[1] > 0 and abs(point[0]) <= 30 and abs(point[1]) <= 30 and point[0] <= 30:
    #             point[0] = 0
    #             point[1] = 0
    #             check = True
    #         elif point[0] > 0 and point[1] < 0 and max(point[0], src.shape[1]) - src.shape[1] <= 30 and abs(min(point[1], -src.shape[0])) - src.shape[0]  <= 30:
    #             point[0] = min(point[0], src.shape[1]-1)
    #             point[1] = max(point[1], -src.shape[0]+1)
    #             check = True
            
    #         # if check == True:
    #             # print(point)
    #         temp_corners.append(point)
    # print("checking init corner")
    # temp_init = cdstP.copy()
    # for temp in temp_corners:
    #     # cv.circle(temp_init,(int(temp[0]), abs(int(temp[1]))), 1, (0,0,255), -1)
    #     # if ori2[abs(int(temp[1])) , abs(int(temp[0]))] > 0:
    #     print(temp, check_nearCorner(temp, temp_corners))
    #     if findNearestWhite(ori2, temp) < 5:
    #         corners.append(temp)
    # print(cdstP.shape)
    

    # print(len(corners))
    
    # # hull = cv.convexHull(np.array(corners, dtype=np.int32))
    # # print(hull)
    # # temp = corners
    # # corners = []
    # print("*********")
    # # if len(temp) > 6:
    # #     for i in range(len(temp)):
    # #         print(ori2[int(temp[i][1]), abs(int(temp[i][0]))])
    # #         if ori2[int(temp[i][1]), abs(int(temp[i][0]))] > 0:
    # #             corners.append(temp[i])
    # # temp = corners
    # # corners = []
    # # for tp in temp:
    # #     if src[abs(int(tp[1])),int(tp[0])] > 0:
    # #         corners.append(tp)
    # print("*********")
    # print(len(corners))
    
    # print(len(corners))
    # temp_corners = corners
    # corners = []
    # for corner in temp_corners:
    #     corner[1] = abs(corner[1])
    #     # if ori2[int(corner[1]), int(corner[0])] == 0:
    #     #     cv.circle(ori2,(int(corner[0]), int(corner[1])), 1, (0,0,255), -1)
    #     # print(corner, findNearestWhite(src, corner) )
    #     if findNearestWhiteRemove(src, corner) < 5:
    #         corners.append(corner)
    #         cv.circle(temp_init,(int(corner[0]), int(corner[1])), 3, (0,0,255), -1)
    # cv.imshow("src", src)
    # cv.imshow("temp_init", temp_init)
    # # cv.waitKey(0)
    #     # 
    # print(corners)
    # # if len(corners) > 6:
    # print("grouping points: before")
    # print(len(corners))
    # temp = corners
    # corners = []
    # dict_label = {}
    # flag = [0]*len(temp)
    # # for i in range(len(temp)):
    # #     dict_label[tuple(temp[i])] = 0
    # for i in range(len(temp)):

    #     # print(src[int(temp[i][1]), int(abs(temp[i][0]))])
    #     for j in range(i+1, len(temp)):
    #         dis = distancePoint(temp[i], temp[j])
    #         # print(dis)
    #         if dis <= gp_thres:
    #             # if tuple(temp[i]) in dict_label.keys():
                    
    #             #     flag[j] = dict_label[tuple(temp[i]) ]
    #             # else:
    #             #     dict_label[tuple(temp[i])] = i+1
    #             #     dict_label[tuple(temp[j])] = i+1
    #                 # flag[i] = i+1
    #             if j == 12:
    #                 print(dis, i)
    #             flag[i] = i+1
    #             flag[j] = i+1

    # print(flag)
    # for i in range(len(temp)):
    #     print(i , flag[i])
    #     if flag[i] == 0:
            
    #         corners.append(temp[i])
    #         # print(i)
    #     elif flag[i] < 1e9:
    #         cnt = 1
    #         p = temp[i]
    #         # print(i)
    #         # print(p)
            
    #         for j in range(i + 1, len(temp)):
    #             if flag[i] == flag[j]:
    #             # if dict_label[tuple(temp[i])] == dict_label[tuple(temp[j])]:
    #                 print(i, j)
    #                 p[0] += temp[j][0]
    #                 p[1] += temp[j][1]
    #                 cnt += 1
    #                 flag[j] = 1e9
    #         # print(p)
    #         p[0] = int(p[0]/cnt)
    #         p[1] = int(p[1]/cnt)
    #         flag[i] = 1e9
    #         # print(flag)
    #         corners.append(p)
    # print(len(corners))
    # print(corners)
    # # temp  = corners
    # # corners = []
    # # for tp in temp:
    # #     print(findNearestWhite(ori2, tp) , check_nearCorner(tp, temp) )
    # #     if findNearestWhite(ori2, tp) < 5 and check_nearCorner(tp, temp) > 10:
    # #         corners.append(tp)
    # corners = find_hull(corners)
    
    # tem_gr = ori.copy()
    # tem_gr = cv.cvtColor(tem_gr,cv.COLOR_GRAY2BGR)
    
    # for corner in corners:
    #     print(corner)
    #     cv.circle(tem_gr,(int(corner[0]), int(corner[1])), 4, (0,0,255), -1)   
    
    # cv.imshow('tem_gr', tem_gr)
    # # cv.waitKey(0)
    
    
    # print(corners)
    temp = corners_shape
    temp.append(corners_shape[0])
    temp.append(corners_shape[1])
    corners = []
    # print("checking same line", len(temp))
    # print(temp)
    arr_same_line = []
    temp_line = ori.copy()
    temp_line = cv.cvtColor(temp_line, cv.COLOR_GRAY2BGR)
    for i in range(len(temp)-1):
        
        try:
            # print(ori2[temp[i][1],temp[i][0]])
            if ori2[temp[i][1],temp[i][0]] >= 0:
                if i + 2 < len(temp):
                    a = temp[i]
                    b = temp[i+1]
                    c = temp[i+2]
                    
                    # print(abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))), a, b, c)
                    if abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))) <=  0.02*src.shape[0]*src.shape[1]:
                        # print(abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))), a, b, c)
                        cv.circle(temp_line,(int(a[0]), int(a[1])), 4, (0,0,255), -1)
                        cv.circle(temp_line,(int(b[0]), int(b[1])), 4, (0,0,255), -1)
                        cv.circle(temp_line,(int(c[0]), int(c[1])), 4, (0,0,255), -1)
                    # if abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))) <=  500:
                        # print(i)
                        if len(arr_same_line) > 0:
                            a1 = arr_same_line[0]
                            b1 = temp[i+1]
                            c1 = temp[i+2]
                            if abs((a1[0]*(b1[1]-c1[1]) + b1[0]*(c1[1]-a1[1]) + c1[0]*(a1[1]-b1[1]))) >  0.02*src.shape[0]*src.shape[1]:
                                corners.append(arr_same_line[0])
                                
                                arr_same_line = [temp[i+1]]
                                # print("check Huong ", arr_same_line, corners)
                        else:
                            arr_same_line.append(temp[i])
                        # print(arr_same_line, corners)
                    else:
                        # print(len(arr_same_line), arr_same_line)
                        if len(arr_same_line) > 0:
                            corners.append(arr_same_line[0])
                            arr_same_line = []
                        else:
                            
                            # print(i)
                            corners.append(temp[i])
                else:
                    if len(arr_same_line) > 0:
                        corners.append(arr_same_line[0])
                        arr_same_line = []
                    else:
                        corners.append(temp[i])
            else:
                # print("point out", temp[i])     
                cv.circle(ori2,(int(temp[i][0]), int(temp[i][1])), 4, (10,90,255), -1)      
        except:
            print("point out", temp[i])
    # print(len(corners_shape))
    # print(corners_shape)
    temp = corners_shape
    corners = []
    dict_label = {}
    flag = [0]*len(temp)
    # for i in range(len(temp)):
    #     dict_label[tuple(temp[i])] = 0
    for i in range(len(temp)):

        # print(src[int(temp[i][1]), int(abs(temp[i][0]))])
        for j in range(i+1, len(temp)):
            dis = distancePoint(temp[i], temp[j])
            # print(dis)
            if dis <= gp_thres:
                # if tuple(temp[i]) in dict_label.keys():
                    
                #     flag[j] = dict_label[tuple(temp[i]) ]
                # else:
                #     dict_label[tuple(temp[i])] = i+1
                #     dict_label[tuple(temp[j])] = i+1
                    # flag[i] = i+1
                # if j == 12:
                #     print(dis, i)
                flag[i] = i+1
                flag[j] = i+1

    # print(flag)
    for i in range(len(temp)):
        # print(i , flag[i])
        if flag[i] == 0:
            
            corners.append(temp[i])
            # print(i)
        elif flag[i] < 1e9:
            cnt = 1
            p = temp[i]
            # print(i)
            # print(p)
            
            for j in range(i + 1, len(temp)):
                if flag[i] == flag[j]:
                # if dict_label[tuple(temp[i])] == dict_label[tuple(temp[j])]:
                    print(i, j)
                    p[0] += temp[j][0]
                    p[1] += temp[j][1]
                    cnt += 1
                    flag[j] = 1e9
            # print(p)
            p[0] = int(p[0]/cnt)
            p[1] = int(p[1]/cnt)
            flag[i] = 1e9
            # print(flag)
            corners.append(p)
    # cv.imshow("temp_line", temp_line)
    temp_corners = corners.copy()
    temp_corners.append(temp_corners[0])
    gr_lines = []
    dict_line = {}
    for i in range(len(temp_corners) - 1):
        gr_lines.append(find_line(temp_corners[i][0], -temp_corners[i][1], temp_corners[i+1][0], - temp_corners[i+1][1]))
        if tuple(temp_corners[i]) not in dict_line.keys():
            dict_line[tuple(temp_corners[i])] = [find_line(temp_corners[i][0], -temp_corners[i][1], temp_corners[i+1][0], - temp_corners[i+1][1])]
        else:
            dict_line[tuple(temp_corners[i])].append(find_line(temp_corners[i][0], -temp_corners[i][1], temp_corners[i+1][0], - temp_corners[i+1][1]))
        if tuple(temp_corners[i+1]) not in dict_line.keys():
            dict_line[tuple(temp_corners[i+1])] = [find_line(temp_corners[i][0], -temp_corners[i][1], temp_corners[i+1][0], - temp_corners[i+1][1])]
        else:
            dict_line[tuple(temp_corners[i+1])].append(find_line(temp_corners[i][0], -temp_corners[i][1], temp_corners[i+1][0], - temp_corners[i+1][1]))
    print("dict line")
    print(dict_line)
    for corner in corners:
    
        cv.circle(cdstP,(int(corner[0]), int(corner[1])), 4, (0,0,255), -1)
    linesP_color_temp = []
    print(linesP_color)
    # for i in range(len(gr_lines)):
    #     j = 0
        # print(gr_lines[i])
        # for line in linesP_color:
    if linesP_color is not None:
        for k in range(0, len(linesP_color)):
            
            l = linesP_color[k][0]
            
            find = find_line(l[0], -l[1], l[2], -l[3])
            dis = 1e9
            index = 0
            idx_cor = 0
            for corner in corners:
                # check = False
                # dis = min(dis, check_line_P(find, (corner[0], -corner[1])))
                if dis > check_line_P(find,(corner[0], -corner[1])):
                    dis = check_line_P(find,(corner[0], -corner[1]))
                    idx_cor = index

                index = index + 1
            # print(k, dis, find)
            if dis <= 10:
                # print(dis, idx_cor, corners[idx_cor])
                # a = [0, 0]
                # b = [0, 0]
                if idx_cor == len(corners) - 1:
                    a = corners[0]
                    b = corners[idx_cor - 1]
                else:
                    a = corners[idx_cor + 1]
                    b = corners[idx_cor - 1]
                check = False
                # print(dis, idx_cor, corners[idx_cor], a, b)
                for line in dict_line[tuple(a)]:
                    if check_line_extra(line, find, thresh):
                        check = True
                for line in dict_line[tuple(b)]:
                    if check_line_extra(line, find, thresh):
                        check = True
                # print(dis, idx_cor, check)
                if check:
                    color = list(np.random.random(size=3) * 256)
                    cv.line(temp_find, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
                    # cv.imwrite("test_line/" + str(j) + "extra.jpg", temp)
                    j = j + 1
                    linesP_color_temp.append(linesP_color[k])
                # if check_line_extra(gr_lines[i], find, thresh) and dis <= 10:
                #     # print(l)
                #     # temp = cdstP.copy()
                #     color = list(np.random.random(size=3) * 256)
                #     cv.line(temp_find, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
                #     # cv.imwrite("test_line/" + str(j) + "extra.jpg", temp)
                #     j = j + 1
                #     linesP_color_temp.append(linesP_color[k])
                    # ex_tra_line.append(find)
                    # ex_tra_pt.append(l)
    linesP_color = linesP_color_temp
    # print(len(linesP_color))
    # print(linesP_color)
    cv2.imshow("temp_find inner line", temp_find)
    # cv.waitKey(0)
    print(corners)
    if len(corners) == 4:
        
        check_hull = find_hull(corners)
        a = check_hull[0]
        b = check_hull[1]
        c = check_hull[2]
        d = check_hull[3]

        line1 = find_line(a[0], -a[1], b[0], -b[1])
        line2 = find_line(d[0], -d[1], c[0], -c[1])

        
        line3 = find_line(b[0], -b[1], c[0], -c[1])
        line4 = find_line(a[0], -a[1], d[0], -d[1])

        if check_line_parralell(line1, line2) and check_line_parralell(line3, line4):

            count = np.array(corners, dtype=np.int32)           
            hull = cv.convexHull(count)
            
            hull_list = [hull]
            for i in range(len(hull_list)):
                mask = np.zeros(src.shape[:2], dtype="uint8")
                color = list(np.random.random(size=3) * 256)
                cv.drawContours(mask, hull_list , i, color, -1)
                        
                # cv.imshow("mask", mask)
                cropt = cv.bitwise_and(src_color, src_color, mask=mask)
                # cv.imshow("mask cropt "+ str(i), cropt)
                name = f.split('.')
                # print("rec")
                cv.imwrite("D:/IU_Studying/research/AIC/Auto-retail-syndata-release/rectangle/label/"+ name[0]+"_mask.jpg", mask)
                cv.imwrite("D:/IU_Studying/research/AIC/Auto-retail-syndata-release/rectangle/seg/"+ name[0]+"_mask.jpg", cropt)
        else:
            # print(line1)
            # print(line2)
            # print(line3)
            # print(line4)
            print("no rec")
        # contours = [np.array(corner, dtype=np.int32)]
        #     # contours.append()
        # for i in range(len(contours)):
        #     mask = np.zeros(src.shape[:2], dtype="uint8")
        #     color = list(np.random.random(size=3) * 256)
        #     cv.drawContours(mask, contours , i, color, -1)
                    
        #     # cv.imshow("mask", mask)
        #     cropt = cv.bitwise_and(src_color, src_color, mask=mask)
        #     cv.imshow("mask cropt "+ str(i), cropt)
        pass
    elif len(corners) == 5 or len(corners) == 6:
        print("----------------------------")
        image_temp = src_color.copy()
        line_candidates = []
        if linesP_color is not None:
            # print(len(linesP_color))
            for i in range(0, len(linesP_color)):
                l = linesP_color[i][0]
                # print(i, l)
                temp = cdstP.copy()
                color = list(np.random.random(size=3) * 256)


                # cv.line(temp, (l[0], l[1]), (l[2], l[3]), color, 3, cv.LINE_AA)
                line_oxy = find_line(l[0], -l[1], l[2], -l[3])
                # cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), color, 3, cv.LINE_AA)
                check_line_cnt = False
                for corner in corners:
                    value = check_line_P(line_oxy, (corner[0], -corner[1]))
                    # print(value, corner)
                    if abs(value) <= 350:
                        # cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), color, 3, cv.LINE_AA)
                        check_line_cnt = True
                        # print(check_line_cnt)
                if check_line_cnt == True:
                    # print("111111")
                    line_candidates.append(line_oxy)
        # print("-----------------------", len(corners))
        # print(corners)
        # print(len(line_candidates))
        dict_cnt = {}
        line_cnt = {}
        # print("------------****-----------")
       
        
        aprox = corners
        aprox.append(corners[0])
        for i in range(len(aprox)):
            if i+2 < len(aprox):
                a = [abs(int(aprox[i][0])), int(-aprox[i][1])] 
                b = [abs(int(aprox[i+1][0])), int(-aprox[i+1][1])]
                c = [abs(int(aprox[i+2][0])), int(-aprox[i+2][1])]
                cv.circle(image_temp,(int(a[0]), -int(a[1])), 3, (0,0,255), -1)
                cv.circle(image_temp,(int(b[0]), -int(b[1])), 3, (0,0,255), -1)
                cv.circle(image_temp,(int(c[0]), -int(c[1])), 3, (0,0,255), -1)
                arr = [a, b, c]
                ps = findMissingPoint(a, b, c)
                for p1 in ps:
                    # print("missing ", a, b, c, p1)
                    if p1 is not None:
                        
                        # print("misss", p1)
                        try:
                            cv.circle(image_temp,(int(p1[0]), -int(p1[1])), 3, (0,0,255), -1)
                            # if ori[-p1[1], p1[0]] != 255 or check_nearCorner(p1, corners) < 10:
                            if ori[-p1[1], p1[0]] != 255:
                                # print(p1)
                                p1 = [1000, 1000]
                                
                            # cv.circle(cdstP,(int(p2[0]), -int(p2[1])), 3, (25,100,255), -1)
                            # cv.circle(cdstP,(int(p3[0]), -int(p3[1])), 3, (25,100,255), -1)
                            else:
                                cv.circle(image_temp,(int(p1[0]), -int(p1[1])), 3, (25,100,255), -1)
                                for line in line_candidates:
                                    value1 = abs(check_line_P(line, p1))
                                    # value2 = abs(check_line_P(line, p2))
                                    # value3 = abs(check_line_P(line, p3))
                                    # print(value1, value2, value3, p1, p2, p3)
                                    # print(value1, arr, p1)
                                    if tuple(p1) in dict_cnt.keys():
                                        if value1 < dict_cnt[tuple(p1)]:

                                            dict_cnt[tuple(p1)] = min(value1, dict_cnt[tuple(p1)] )
                                            line_cnt[tuple(p1)] = arr

                                    else:
                                        dict_cnt[tuple(p1)] = value1  
                                        line_cnt[tuple(p1)] = arr
                        except:
                            p1 = [1000, 1000]
        cv.imshow("image_temp", image_temp)        
                




        import operator
        if len(dict_cnt) > 0:
            # print(dict_cnt)
            sort_orders = sorted(dict_cnt.items(), key=lambda x: x[1])
            print(sort_orders)
            # p = float(list(sort_orders[0])[0])
            cv.circle(cdstP,(int(list(sort_orders[0])[0][0]), abs(int(list(sort_orders[0])[0][1]))), 4, (88,100,0), -1)
            selected_point = select_best_inside(sort_orders, line_cnt, corners)
            line_chosen = line_cnt[(selected_point[0][0], selected_point[0][1])]
            # print(line_chosen)
            point_start = line_chosen[0]
            # print(point_start)
            # value = check_line_P(line_chosen, (corners[0][0], -corners[0][1]))
            index_start = 0 
            # for i in range(1, len(corners)):
            #     value_temp = check_line_P(line_chosen, (corners[i][0], -corners[i][1]))
            #     print(value_temp, value, corners[i])
            #     if value_temp < value:
            #         value = value_temp
            #         point_start = corners[i]
            #         index_start = i
                    
            
            cv.circle(cdstP,(int(point_start[0]), int(abs(point_start[1]))), 4, (88,100,90), -1)    
            # corners.append([list(sort_orders[0])[0][0], -list(sort_orders[0])[0][1]])
            
            final_corner = []
            for corner in corners:
                final_corner.append( [int(corner[0]), abs(int(corner[1]))])
            # print(len(final_corner))
            print("------------------")
            print("---------********************---------")
            print(selected_point)
            # print(final_corner)
            inside = [int(list(selected_point)[0][0]), abs(int(list(selected_point)[0][1]))]
            # print(inside)
            count = []
            
            count = np.array(final_corner, dtype=np.int32)           
            hull = cv.convexHull(count)
            # print(hull.shape)
            # print(hull[0][0][0], hull[0][0][1])
            # print(hull)
            arr = []
            for i in range(hull.shape[0]):
                if int(point_start[0]) == hull[i][0][0] and int(point_start[1]) == -hull[i][0][1]:
                    index_start = i
                    point_start = [hull[i][0][0], hull[i][0][1]]
                arr.append([hull[i][0][0], hull[i][0][1]])
            
            # print(point_start, index_start)
            index = index_start + 1
            slice_corner = [point_start]
            if index == len(arr):
                index = 0
            while index != index_start:
                # print(index)
                
                slice_corner.append(arr[index])
                index = index + 1
                if index == len(arr):
                    index = 0
            # print(corners)
            slice_corner.append(point_start)
            # print(slice_corner)
            gr_corners = [slice_corner[0]]
            box_shapes = []
            index = 1
            while index < len(slice_corner):
            # for i in range(1, len(slice_corner)):
                # print(index)
                if len(gr_corners) == 2:
                    # print("checking shape")
                    gr_corners.append(slice_corner[index])
                    gr_corners.append(inside)
                    # print(gr_corners)
                    hulls = find_hull(gr_corners)
                    if len(hulls) == 4:
                        box_shapes.append(gr_corners)
                        # print(gr_corners)
                        # print(check_shape(gr_corners))
                        gr_corners = [slice_corner[index]]
                        index = index + 1
                    else:
                        
                        index = index - 1
                        # print(i)
                        gr_corners= [slice_corner[index]]
                        index = index + 1
                        # print(gr_corners)
                else:
                    gr_corners.append(slice_corner[index])
                    index = index + 1


            # print(box_shapes)
            contours = []
            for box in box_shapes:
                # print(box)
                contours.append(np.array(box, dtype=np.int32))
            for i in range(len(box_shapes)):
                mask = np.zeros(src.shape[:2], dtype="uint8")
                color = list(np.random.random(size=3) * 256)
                cv.drawContours(mask, contours , i, color, -1)
                        
                # cv.imshow("mask", mask)
                cropt = cv.bitwise_and(src_color, src_color, mask=mask)
                cv.imshow("mask cropt "+ str(i), cropt)
                name = f.split('.')
                cv.imwrite('D:/IU_Studying/research/AIC/Auto-retail-syndata-release/box shape/label/'+ name[0] +"_mask_" + str(i) + ".jpg", mask)
                cv.imwrite('D:/IU_Studying/research/AIC/Auto-retail-syndata-release/box shape/seg/'+ name[0] +"_seg_" + str(i) + ".jpg", cropt)
        # pot = list(sort_orders.keys())[0]
        # cv.circle(cdstP,(int(pot[0]), abs(int(pot[1]))), 3, (10,156,255), -1)
    # kernel1 = np.ones((5,5),np.uint8)
    # cdstP = cv.dilate(cdstP,kernel1,iterations = 2)
    # cdstP = cv.erode(cdstP,kernel1,iterations = 3)

    # dst = cv.cornerHarris(cdstP,2,3,0.02)

    else:
        name = f.split('.')
        src_color = cv.imread(cv.samples.findFile(file_color+f))
        default_file = file +name[0] +'_seg.' + name[1]
        src = cv.imread(cv.samples.findFile(default_file))
        
        src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        src[src>0]=255
        cropt = cv.bitwise_and(src_color, src_color, mask=src)
        cv.imshow("mask cropt "+ str(i), cropt)
        cv.imwrite('D:/IU_Studying/research/AIC/Auto-retail-syndata-release/box shape/failed/'+name[0]+'.jpg', cropt)
    # Threshold for an optimal value, it may vary depending on the image.
    # ori[dst>0.4*dst.max()]=[0,0,255]
    cv.imshow("src", src)
    # cv.imshow("Source 1", ori)
    # cv.imshow("Source 2 ", ori2)
    cv.imshow("src_color", src_color)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    # from PIL import Image, ImageFilter
    
    # img = Image.open('Auto-retail-syndata-release/syn_image_train/00001_1.jpg')
    
    # # Converting the image to grayscale, as Sobel Operator requires
    # # input image to be of mode Grayscale (L)
    # img = img.convert("L")
    
    # # Calculating Edges using the passed laplican Kernel
    # final = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
    #                                         -1, -1, -1, -1), 1, 0))
    
    # final.save("EDGE_sample.png")
    cv.waitKey(1)
    cv.destroyAllWindows()
    return 0




if __name__ == "__main__":
    import os
    path = 'dataset/train'
    path_seg = 'dataset/segmentation_labels/'
    files = os.listdir(path)
    # fileOther = open("dataset/label.txt", "r")
    # others = []
    # for x in fileOther:
    #     others.append(int(x))
    # files = ['00001_104768.jpg','00001_1.jpg','00001_117.jpg','00004_83269.jpg', '00004_96187.jpg', '00001_813.jpg', '00116_111251.jpg', '00001_7422.jpg', '00001_1589.jpg', '00087_93973.jpg']
    # files = ['00002_34667.jpg', '00002_35479.jpg', '00002_36291.jpg', '00002_60509.jpg', '00002_65381.jpg', '00002_101405.jpg', '00003_1047.jpg', '00005_5305.jpg',
    #         '00005_32930.jpg', '00005_33278.jpg', '00005_50183.jpg', '00009_79214.jpg', '00009_69216.jpg', '00009_86599.jpg', '00014_106709.jpg', '00015_46249.jpg',
    #         '00015_105943.jpg', '00016_12534.jpg', '00016_15782.jpg', '00016_20770.jpg', ]
    # files = ['00002_32115.jpg']
    # main(sys.argv[1:], f, path_seg, path)
    for f in files:
        # try:
        print(f)
    # main(sys.argv[1:], f, path_seg, path)
        name = f.split('.')
        src_color = cv.imread(cv.samples.findFile(path+f))
        default_file = path_seg +name[0] +'_seg.' + name[1]
        src = cv.imread(cv.samples.findFile(default_file))
        corners_check_shape, corners = check_shape(src)
        print(corners)
        lalelnum = int(f.split('_')[0])
        # if lalelnum >= 1:
        #     isOther = False
        #     for other in others:
        #         if lalelnum == other:
        #             isOther = True
        #     isOther = False
        if corners_check_shape > 7:
        # if isOther :
            name = f.split('.')
            src_color = cv.imread(cv.samples.findFile(path+f))
            default_file = path_seg + name[0] +'_seg.' + name[1]
            src = cv.imread(cv.samples.findFile(default_file))
            
            src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            # src[src>128]=255
            _, src = cv2.threshold(src, 128, 1, cv2.THRESH_BINARY)
            cropt = cv.bitwise_and(src_color, src_color, mask=src)
            # cv.imshow("mask cropt "+ str(lalelnum), cropt)
            src *= 255
            # cv.imshow("mask cropt mask"+ str(lalelnum), src)
            
            cv.imwrite('dataset/other/label/'+name[0]+'_mask.jpg', src)
            cv.imwrite('dataset/other/seg/'+name[0]+'_seg.jpg', cropt)
            # cv.waitKey(0)
            cv.destroyAllWindows()
            print('other')
        elif corners_check_shape == 4:
            print("444")
            name = f.split('.')
            src_color = cv.imread(cv.samples.findFile(path+f))
            default_file = path_seg +name[0] +'_seg.' + name[1]
            src = cv.imread(cv.samples.findFile(default_file))
            
            src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            # src[src>128]=255
            _, src = cv2.threshold(src, 128, 1, cv2.THRESH_BINARY)
            cropt = cv.bitwise_and(src_color, src_color, mask=src)
            src *= 255
            # cv.imshow("mask cropt "+ str(lalelnum), cropt)
            # cv.imshow("mask cropt mask"+ str(lalelnum), src)
            cv.imwrite("dataset/rectangle/label/"+ name[0]+"_mask.jpg", src)
            cv.imwrite("dataset/rectangle/seg/"+ name[0]+"_mask.jpg", cropt)
            # cv.waitKey(0)
            # pass
        else:
            print(f)
            
            # main(sys.argv[1:], f, path_seg, path, corners)
            # cv.waitKey(1)

        # except:
        #     print(f, " error")
    
    
    
            