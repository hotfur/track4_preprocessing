import rectangle as R
import random
import math
import cv2
import numpy as np
from rectangle import Point

"""
    Randomly erase in the shape of finger to an img 
"""
def randomFinger(img, p=0.5, scale = (0.125, 0.33), ratio=(0.125, 0.75), 
                  value = (0,0,0), inplace = False, min_count = 1, max_count = 3):
    # probability to execute the random
    if random.random()>p:
        return img
    
    # initial image values
    res = np.array(img)
    img_h = img.shape[0]
    img_w = img.shape[1]
    img_ch = img.shape[2]
    
    img_area = img_h*img_w
    
    # base on number of finger in range
    count = min_count if min_count == max_count else \
            random.randint(min_count, max_count)
    
    # each finger
    for _ in range(count):
        #try 10 times for sure
        for attempt in range(10):
            # choose target area
            target_area = random.uniform(*scale) * img_area / count
            
            # choose target ratio
            aspect_ratio = math.exp(random.uniform(*ratio))
            
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if w < img_w and h < img_h:
                rect = R.Fourgon(Point(0,0), Point(w, 0), Point(w, h), Point(0, h))
                # choose random translate
                translate_x = random.randint(int(0),int(img_w))
                translate_y = random.randint(int(0),int(img_h))
                # choose random rotation
                theta = random.randint(0,360)
                center = R.Point(w/2, h/2)
                
                rect.rotate(center, theta)
                rect.translate(translate_x, translate_y)
                
                # fill the rectangle with points
                res = cv2.fillConvexPoly(res, 
                                   np.array(list((list((rect.p1.x, rect.p1.y)),
                                                         list((rect.p2.x, rect.p2.y)), 
                                                         list((rect.p3.x, rect.p3.y)),
                                                         list((rect.p4.x, rect.p4.y)))), np.int32), 
                                   value)
                
                # fill the (half) circle
                res = cv2.circle(res, 
                           (int((rect.p3.x+rect.p4.x)/2), int((rect.p3.y+rect.p4.y)/2)),
                           int(w/2),
                           value,
                           -1
                           )
                break
    if inplace:
        img = res
    return res