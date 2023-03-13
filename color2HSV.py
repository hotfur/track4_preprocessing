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

path = '../dataset/backup/abstract/test/'
path_seg = '../dataset/backup/abstract/test_seg/'
files = os.listdir(path)

for i in range(len(files)):
    f = "00036_1508.jpg"
    name = f.split('.')
    print(f)
    img_RGB = cv2.imread(path + f)
    img_seg = cv2.imread(path_seg + f)
    img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
    _, img_seg = cv2.threshold(img_seg, 128, 1, cv2.THRESH_BINARY)

    img_Luv = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2Lab)
    masked = np.ma.array(data=img_Luv, mask=np.repeat(np.logical_not(img_seg)[..., None], 3, axis=-1), fill_value=0)
    masked[:,:,0] = np.average(img_Luv[:,:,0], weights=img_seg)

    img_Luv = cv2.cvtColor(masked, cv2.COLOR_Lab2RGB)

    titles = ['With L channel', 'With u channel', 'With v channel']
    cmaps = [plt.cm.Greys, plt.cm.Greys, plt.cm.Greys]
    fig, axes = plt.subplots(1, 4, figsize=(10, 30), num=name[0])
    objs = zip(axes[:-1], (img_Luv[:, :, 0] / 255.0, img_Luv[:, :, 1] / 255.0, img_Luv[:, :, 2] / 255.0), titles, cmaps)

    for ax, channel, title, cmap in objs:
        ax.imshow(channel, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(())
        ax.set_yticks(())

    axes[3].imshow(img_Luv)
    axes[3].set_title('Original')

    plt.show()
