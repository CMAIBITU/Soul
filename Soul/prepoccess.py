#分文件夹存储
import os,sys
import pandas
import openpyxl
import numpy as np
import shutil
import json
import cv2

# 打开文件
path = "D:\dataset\OCTANEW\OCTA_SD_Feng"


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
dirs = os.listdir( path )
print(dirs)
data = pandas.DataFrame(dirs)
#data.to_excel("peoplenames.xlsx")
filess = []
for dir in dirs:
    path = os.path.join(r"D:\dataset\OCTANEW\OCTA_SD_Feng",dir)
    print(path)
    dirs1 = os.listdir(path)
    for dir1 in dirs1:
        dir1 = os.path.join(path, dir1)
        filess.append(dir1)
    for file in filess:
        str = file.split("_")
        str.remove(str[0])
        str.remove(str[0])
        str.remove(str[0])
        str.remove(str[2])
        str2 = str[-1]
        s = '.jpg'
        s1 = str2.rstrip(s)
        str[-1] = s1
        print(str)
        name = str[2]
    path1 = os.path.join(path, name)
    mkdir(path1)
    # with open(textfile, "a") as f_obj:
    #     f_obj.write("\n")
    #     json.dump(str, f_obj)
#编号分类
import os,sys
import pandas
import openpyxl
import numpy as np
import shutil
import json
import cv2

# 打开文件
path = "D:\dataset\OCTANEW\OCTA_SD_Feng"


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
dirs = os.listdir( path )
print(dirs)
data = pandas.DataFrame(dirs)
#data.to_excel("peoplenames.xlsx")
filess = []
for dir in dirs:
    path = os.path.join(r"D:\dataset\OCTANEW\OCTA_SD_Feng",dir)
    print(path)
    dirs1 = os.listdir(path)
    for dir1 in dirs1:
        dir1 = os.path.join(path, dir1)
        filess.append(dir1)
    for file in filess:
        str = file.split("_")
        str.remove(str[0])
        str.remove(str[0])
        str.remove(str[0])
        str.remove(str[2])
        str2 = str[-1]
        s = '.jpg'
        s1 = str2.rstrip(s)
        str[-1] = s1
        print(str)
        name = str[2]
    path1 = os.path.join(path, name)
    mkdir(path1)
    # with open(textfile, "a") as f_obj:
    #     f_obj.write("\n")
    #     json.dump(str, f_obj)
#提取ROI
import os, sys
import pandas
import openpyxl
import numpy as np
import shutil
import json
import cv2
import re
import shutil

# 打开文件
path = "C:/Users/30488/XJY/DATA/OCTA_SD68-1/OCTA-SD68-1/twice"

# 生成文件夹
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
path1 = "C:/Users/30488/XJY/DATA/OCTA_SD68-1/OCTA-SD68-1/twice/R"
path2 = "C:/Users/30488/XJY/DATA/OCTA_SD68-1/OCTA-SD68-1/twice/V/Superficial"
path3 = "C:/Users/30488/XJY/DATA/OCTA_SD68-1/OCTA-SD68-1/twice/V/Deep"
path4 = "C:/Users/30488/XJY/DATA/OCTA_SD68-1/OCTA-SD68-1/twice/V/Outer Retina"
path5 = "C:/Users/30488/XJY/DATA/OCTA_SD68-1/OCTA-SD68-1/twice/V/Choriocapillaris"
# 根据文件夹产生目录列表
dirs = os.listdir(path)
# data = pandas.DataFrame(dirs)
filess = []
filess1 = []
for dir in dirs:
    path = os.path.join(r"C:/Users/30488/XJY/DATA/OCTA_SD68-1/OCTA-SD68-1/twice/",dir)
    dirs1 = os.listdir(path)
    for dir1 in dirs1:
        dir1 = os.path.join(path,dir1)
        filess.append(dir1)
mkdir(path1)
mkdir(path2)
mkdir(path3)
mkdir(path4)
mkdir(path5)
num6 = 1
num1 = 1
num2 = 1
num3 = 1
num4 = 1
num5 = 1
for file in filess:
    str2 = file.split("_")
    print(str2)
    str3 = str2[1].rstrip("\\").split("/")
    print(str3)
    if "HD Angio Retina" in str2 or "Angio Retina" in str2:
        img = cv2.imread(file)
        img1 = img[412:668, 23:724]
        img2 = img[412:668, 744:1445]
        img3 = img[40:386, 22:367]
        img4 = img[40:386, 382:728]
        img5 = img[40:386, 742:1088]
        img6 = img[40:386, 1101:1447]
        cv2.imwrite(path1 + "\\" + str3[3] + "_" + str2[5] + "_" + str(num5) + ".jpg", img1)
        num5= num5 + 1
        cv2.imwrite(path1 + "\\" + str3[3] + "_" + str2[5] + "_" + str(num5) + ".jpg", img2)
        num5 = num5 + 1
        cv2.imwrite(path2 + "\\" + str3[3] + "_" + str2[5] + "_" + str(num2) + ".jpg", img3)
        num2 = num2 + 1
        cv2.imwrite(path3 + "\\" + str3[3] + "_" + str2[5] + "_" + str(num3) + ".jpg", img4)
        num3 = num3 + 1
        cv2.imwrite(path4 + "\\" + str3[3] + "_" + str2[5] + "_" + str(num4) + ".jpg", img5)
        num4 = num4 + 1
        cv2.imwrite(path5 + "\\" + str3[3] + "_" + str2[5] + "_" + str(num6) + ".jpg", img6)
        num6 = num6 + 1
    elif "Radial Lines" in str2:
        img = cv2.imread(file)
        img1 = img[51:455, 578:1536]
        img2 = img[497:900, 578:1536]
        cv2.imwrite(path1 + "\\" + str3[3] + "_" + str2[5] + "_" + str(num5) + ".jpg", img1)
        num5 = num5 + 1
        cv2.imwrite(path1 + "\\" + str3[3] + "_" + str2[5] + "_" + str(num5) + ".jpg", img2)
        num5 = num5 + 1
    elif "Cross Line" in str2:
        img = cv2.imread(file)
        img1 = img[78:489, 445:1546]
        img2 = img[525:938, 445:1546]
        cv2.imwrite(path1 + "\\" + str3[3] + "_" + str2[5] + "_" + str(num5) + ".jpg", img1)
        num5 = num5 + 1
        cv2.imwrite(path1 + "\\" + str3[3] + "_" + str2[5] + "_" + str(num5) + ".jpg", img2)
        num5 = num5 + 1

    else:
        print(str2)
