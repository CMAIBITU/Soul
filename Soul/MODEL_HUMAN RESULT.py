#融合模型结果
from PIL import  Image
import os
import cv2
from skimage import morphology
import numpy as np
import time
im1 = cv2.imread(r"C:\postgraduate\Biye\code fuxian\结果\AttResU_Net结果\2_11460_1.jpg",0)
im2 = cv2.imread(r"C:\postgraduate\Biye\code fuxian\结果\AttU_Net结果\2_11460_1.jpg",0)
im3 = cv2.imread(r"C:\postgraduate\Biye\code fuxian\结果\OCTA-NET结果\2_11460_1.jpg",0)
im4 = cv2.imread(r"C:\postgraduate\Biye\code fuxian\结果\U-NET结果\2_11460_1.jpg",0)
im5 = cv2.imread(r"C:\postgraduate\Biye\code fuxian\结果\U-NET结果\2_11460_1.jpg",0)
Value = [[0] * im1.shape[1] for i in range(im1.shape[0])]  # 创建一个大小与图片相同的二维数组
for y in range(im1.shape[1]):
    for x in range(im1.shape[0]):
        print(im1[x][y])
        if im1[x][y] == 255:
            Value[x][y] += 1
for y in range(im2.shape[1]):
    for x in range(im2.shape[0]):
        print(im2[x][y])
        if im2[x][y] == 255:
            Value[x][y] += 1
for y in range(im3.shape[1]):
    for x in range(im3.shape[0]):
        print(im3[x][y])
        if im3[x][y] == 255:
            Value[x][y] += 1
for y in range(im4.shape[1]):
    for x in range(im4.shape[0]):
        print(im4[x][y])
        if im4[x][y] == 255:
            Value[x][y] += 1
for y in range(im5.shape[1]):
    for x in range(im5.shape[0]):
        print(im5[x][y])
        if im5[x][y] == 255:
            Value[x][y] += 1
print(Value)
#原图和融合结果叠加
from PIL import Image
import os
import cv2
from skimage import morphology
import numpy as np
import time
path1 = r"C:\Users\30488\XJY\DATA\OCTA_SD_53\1.1\Superficial\ronghe"
path2 = r"C:\Users\30488\XJY\DATA\OCTA_SD_53\1.1\Superficial\image"
list1 = os.listdir(path1)
list2 = os.listdir(path2)
# im = cv2.imread(r"C:\Desktop\labelme\ronghe1\2_11926_10.jpg",1)
# img1 = cv2.imread(r"C:\Desktop\labelme\image\2_11926_10.jpg",1)
for i in list1:
    print(i)
    path3 = os.path.join(path1, i)
    path4 = os.path.join(path2, i)
    print(path4)
    im = cv2.imread(path3, 1)
    img1 = cv2.imread(path4, 1)
    # print(im)
    Value = [[0] * im.shape[1] for i in range(im.shape[0])]  # 创建一个大小与图片相同的二维数组
    for y in range(im.shape[1]):
        for x in range(im.shape[0]):
            rgb = im[x, y]  # 获取一个像素块的rgb
            r = rgb[0]
            g = rgb[1]
            b = rgb[2]
            # if r == 255 and r == 255 and r == 255:  # 判断规则
            # if r != 0 and g  != 0 and b != 0:
            # if r == 255 and g  == 255 and b == 255:
            if r > 190 and g  > 190 and b > 190:
                im[x, y] = (0, 0, 255)
    # print(im)
    img2 = cv2.add(im,img1,0.5)
    # cv2.imshow("hng",im)
    # cv2.imshow("hongse",img2)
    cv2.imwrite(r"C:\Users\30488\XJY\DATA\OCTA_SD_53\1.1\Superficial\label/"+i+".png", img2)
    cv2.waitKey()

