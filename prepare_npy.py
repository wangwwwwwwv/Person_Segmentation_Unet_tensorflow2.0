import os
import cv2
import numpy as np
PATH='C:\\Users\\wsq\\Desktop\\UNET\\LV-MHP-v1'
#test_path=PATH+"\\x_test"
train_path=PATH+'\\x_train'#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

names=os.listdir(train_path)
img_path=train_path+'\\'+names[0][0:8]
img=cv2.imread(img_path)    #!!!!!!!!!!!!!!!!!!!!!!!!!!!
data=np.expand_dims(img,axis=0)
#data=np.expand_dims(data,axis=-1)
print(data.shape)
for name in names[1:]:
    name=name[0:8]
    img_path=train_path+'\\'+name
    img=cv2.imread(img_path)#!!!!!!!!!!!!!!!!!
    img=np.expand_dims(img,axis=0)
    #img=np.expand_dims(img,axis=-1)


    data=np.vstack((data,img))
    print(data.shape)
   
np.save('x_train.npy',data) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print(data.shape)