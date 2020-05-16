# Person_Segmentation_Unet_tensorflow2.0

Using Unet to segmentate LV-MHP-v1 dataset
need to trans the LV-MHP-V1 data to .npy file

1 download dataset(LV-MHP-v1 or others)

2 x\y_trans_png.py   
  base on train.txt /test.txt (filename of train/test data) devide the train/test image and resize  
  set the y_train/y_test data to black and white data

3 prepare_npy.py
  
you might need to understand my code and change some.




inputshape(none，128，128，3)
outputshape(None, 128, 128, 1)

WARNING!!!!

the label(y_train/y_test) is 0/1 
not 0-255
!!!!!!!


QQ 1923269508
wechat: Stephon_Marbury_



