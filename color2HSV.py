import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# main
# def img_process(input_path, output_path, seg_path, color_scale = cv2.COLOR_RGB2HSV):
#     for f in os.listdir(input_path):
#         file = os.path.join(input_path, f)
#         if os.path.isfile(file):
#             name = f.split('.')
#             seg_path = seg_path + name[0] + '_seg.' + name[1]
#             img_RGB = cv2.imread(file)
#
#             img_HSV = cv2.cvtColor(img_RGB, color_scale)
#
#
#             store_name = name[0] + "_HSV." + name[1]
#
#             if not os.path.exists(output_path):
#                 os.makedirs(output_path)
#
#             cv2.imwrite(os.path.join(output_path, store_name), img_HSV)
#         else:
#             img_process(os.path.join(input_path, f), os.path.join(output_path, f))

# img_process('../dataset/train', '../dataset/hsv', '../dataset/segmentation_labels', cv2.COLOR_RGB2HSV)

# loading the file

# path = '../dataset/classify/color_removal/seg/'
# output_path='../dataset/classify/box/test/'
# path_seg = '../dataset/classify/color_removal/label/'
path = '../dataset/an_test/'
files = os.listdir(path)

for f in files:
    if f.__contains__("seg"):
        continue
    name = f.split('.')
    img_RGB = cv2.imread(path + f)
    # img_seg = cv2.imread(path_seg + f)
    img_seg = cv2.imread(path + name[0] + "_seg.jpg")
    img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
    _, img_seg = cv2.threshold(img_seg, 128, 1, cv2.THRESH_BINARY)

    img_Luv = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2Lab)
    # img_Luv[:, :, 0] = img_Luv[:, :, 0] * 0.75
    # img_Luv = cv2.cvtColor(img_Luv, cv2.COLOR_BGR2HSV)
    var = np.var(img_Luv, axis=(0, 1), where=np.repeat(img_seg[..., None], 3, axis=-1).astype(bool))
    mean = np.mean(img_Luv, axis=(0,1), where=np.repeat(img_seg[..., None], 3, axis=-1).astype(bool))
    color_dist = np.sqrt(var[1] + var[2])
    grey_dist = abs(mean[1] - 128) + abs(mean[2]-128)
    # if color_dist > 5:
    print(f)
    print(mean)
    print(grey_dist)
    print(np.sqrt(var))
    print(np.sqrt(var[1]+var[2]))
    print(color_dist < 9.5)
    # print(np.histogram(img_Luv[...,2], weights=img_seg))

    #
    # cv2.imwrite(os.path.join(output_path, f), img_Luv)



    # titles = ['With L channel', 'With u channel', 'With v channel']
    # cmaps = [plt.cm.Greys, plt.cm.Greys, plt.cm.Greys]
    # fig, axes = plt.subplots(1, 4, figsize=(10, 30), num=name[0])
    # objs = zip(axes[:-1], (img_Luv[:, :, 0] / 255.0, img_Luv[:, :, 1] / 255.0, img_Luv[:, :, 2] / 255.0), titles, cmaps)
    #
    # for ax, channel, title, cmap in objs:
    #     ax.imshow(channel, cmap=cmap)
    #     ax.set_title(title)
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #
    # axes[3].imshow(img_Luv)
    # axes[3].set_title('Original')
    #
    # plt.show()
