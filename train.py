import os
import cv2
import numpy as np
from tfUnet import Unet
from tensorflow.compat.v1 import ConfigProto
from tensorflow import keras
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2


image=cv2.imread('4972.jpg')
hight=image.shape[0]
width=image.shape[1]

print(hight,width)
image=cv2.resize(image,(128, 128), interpolation=cv2.INTER_CUBIC)
image=np.expand_dims(image,axis=0)
print(image.shape)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
model = Unet(1)

x_test=np.load('x_test.npy')#980*572*572*3
y_test=np.load('y_test.npy')#980*572*572
x_train=np.load('x_train.npy')#4000*572*572*3
y_train=np.load('y_train.npy')#4000*572*572
y_test=np.where(y_test>0,1,0)
y_train=np.where(y_train>0,1,0)

callback=[
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
]
model.fit(x_train,y_train,batch_size=100,epochs=50,callbacks=callback)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)
#trainset = DataGenerator("membrane/train", batch_size=5)
#model.fit_generator(trainset,steps_per_epoch=5000,epochs=5)
model.save_weights("model.h5")
y=model.predict(image)
np.save('dfs.npy',y)