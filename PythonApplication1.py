import os
import keras
import cv2 as cv
import re
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
datagen = ImageDataGenerator(featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        zca_epsilon=1e-6,
        rotation_range = 40,
        width_shift_range= 0.2,
        height_shift_range = 0.2,
        rescale = 1.0/255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip=True,
        fill_mode = 'nearest',
        
    )
write_path = "C:/Users/bm00133/Desktop/新建文件夹/test/"

def eachFile(filepath):
    count = 0
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath,allDir))
        write_child = os.path.join('%s%s' % (write_path,allDir))
        img = load_img(child)
        nul_num = re.findall(r"\d",child)
        nul_num = int(nul_num[0])
        x = img_to_array(img)
        x = x.reshape((1,)+x.shape)
        i = 0
        for batch in datagen.flow(
        	    x,
        	    batch_size =32,
                shuffle=True,
        	    save_to_dir = write_path,
        	    save_prefix = nul_num,save_format = 'jpg'):

                count += 1
                i += 1
                if i >= 10 :
                       break
    return count

#调用
count = eachFile("C:/Users/bm00133/Desktop/新建文件夹/缺陷样本/")
print ("一共产生了%d张图片"%count)




