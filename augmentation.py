import randomFinger as rF
import numpy as np
import cv2
import random
import math

def affineFinger(img, img_seg, p = 1, degrees = (0, 360), scales = (1, 1.5), shears = (0, 0.5)):
    # probability to execute the random
    if random.random()>p:
        return img, img_seg
    # initial image values
    img_h = img.shape[0]
    img_w = img.shape[1]
    img_ch = img.shape[2]
    
    
    # choose random rotation
    degree = random.randint(*degrees)
    shear_x = random.uniform(*shears)
    shear_y = random.uniform(*shears)
    scale_x = random.uniform(*scales)
    
    
    
    
    # Expand and translate to the center
    shear = np.array([[scale_x, shear_x, 0],
                      [shear_y, scale_x, 0]])
    
    img_w1 = int(img_w+abs(shear[0, 1]*img_h))
    img_h1 = int(img_h+abs(shear[1,0]*img_w))
    
    img_w2 = 2*int(math.sqrt((img_w1*scale_x)**2+(img_h1*scale_x)**2))
    img_h2 = 2*int(math.sqrt((img_w1*scale_x)**2+(img_h1*scale_x)**2))
    
  
    shear[0,2] = -shear[0,1] * img_w1/2 + img_w2*0.4
    shear[1,2] = -shear[1,0] * img_h1/2 + img_h2*0.4
    
    # translation = np.array([[1, 0, ],
    #                         [0, 1, img_h2/3]])
    
    # res = cv2.warpAffine(img, translation, [img_w2, img_h2])
    # res_seg = cv2.warpAffine(img_seg, translation, [img_w2, img_h2])
    # Rescale the image and seg so that the newly scale will sure fit in
    max_len1 = math.sqrt((img_w*scale_x)**2+(img_h*scale_x)**2)
    
    
    res = cv2.warpAffine(img, shear, [img_w2, img_h2])
    res_seg = cv2.warpAffine(img_seg, shear, [img_w2, img_h2])
    
    
    # Rescale again to make sure then rotate
    rotation = cv2.getRotationMatrix2D(center = (img_w2/2, img_h2/2), angle = degree, scale = scale_x)
    
    res = cv2.warpAffine(res, rotation, [img_w2, img_h2])
    res_seg = cv2.warpAffine(res_seg, rotation, [img_w2, img_h2])
    res_seg = rF.randomFinger(res_seg, p=1, scale=(0.05, 0.06))
    return res, res_seg