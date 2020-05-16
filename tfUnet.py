import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.layers import Dropout, Input
#import sklearn

import os
import sys
import time
import tensorflow as tf
from tensorflow import keras



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D,Conv3D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D


def Unet(num_class):
    #inputs = Lambda(lambda x: x / 255) (inputs)
    #----------------------------------------------------------------------------------------第一层，64
    inputs = Input(shape=[128, 128, 3])#输入
    s=Lambda(lambda x:x/255)(inputs)

    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(s)#64核 3*3
    conv1 = Dropout(0.1)(conv1)                                         
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1) #64核 3*3----------conv1第一层输出
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



    #----------------------------------------------------------------------------------------第二层，128
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)#128核 3*3
    conv2 = Dropout(0.1)(conv2)                                        
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)#128核 3*3----------conv2第二层输出
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



    #----------------------------------------------------------------------------------------第三层，256
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool2)#256核 3*3
    conv3 = Dropout(0.2)(conv3)                                       
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv3)#256核 3*3----------conv3第三层输出
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



    #----------------------------------------------------------------------------------------第四层，512
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool3)#512核 3*3
    conv4 = Dropout(0.2)(conv4)                                         
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv4)#512核 3*3----------conv4第四层输出
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  
                       



    #----------------------------------------------------------------------------------------第五层，1024
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool4)#1024核 3*3 
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv5)#1024核 3*3 ，之后开始拼接，反卷积



    #----------------------------------------------------------------------------------------第六层，512
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv5))#反卷积
    up6 = concatenate([conv4,up6], axis = 3)                                              #反卷积之后拼接第四层
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same')(up6)#512核 3*3
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv6)#512核 3*3



    #----------------------------------------------------------------------------------------第七层，256
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))#反卷积2*2
    up7 = concatenate([conv3,up7], axis = 3)                                             #反卷积之后拼接第三层
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same')(up7)#256核 3*3
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv7)#256核 3*3



    #----------------------------------------------------------------------------------------第八层，128
    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    up8 = concatenate([conv2,up8], axis = 3)                                         #反卷积之后拼接第二层   
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same')(up8)#128核 3*3
    conv6 = Dropout(0.1)(conv6)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv8)#128核 3*3


    #----------------------------------------------------------------------------------------最后一层
    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    up9 = concatenate([conv1,up9], axis = 3)                                              
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same')(up9)#64核 3*3    
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv9)#64核 3*3

    outputs = Conv2D(num_class, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = [inputs], outputs = [outputs])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()


    return model
if __name__ == "__main__":
    Unet(1)