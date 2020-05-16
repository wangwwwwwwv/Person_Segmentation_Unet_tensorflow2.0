import os
import cv2
import numpy as np
PATH='C:\\Users\\wsq\\Desktop\\UNET\\LV-MHP-v1'
test_path=PATH+"\\test_list.txt"
train_path=PATH+'\\train_list.txt'
images_path=PATH+'\\images\\'
anno_name=os.listdir(images_path)
fo = open(train_path, "r")
for name in fo.readlines():
    name=name[0:8]
    path_png=images_path+ name
    img=cv2.imread(path_png)
    img=cv2.resize(img,(128, 128), interpolation=cv2.INTER_CUBIC)
    output_path=PATH+'\\x_train\\'+name
    print(output_path)
    cv2.imwrite(output_path,img)

    