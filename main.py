import warnings
warnings.filterwarnings('ignore')
import numpy as np
from keras.models import load_model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping


import os
#import matplotlib.pyplot as plt
import math
import glob
import shutil

root_dir = "D:/datasets/BrainTumorData/Brain Tumor Data Set"
number_of_images={}
for dir in os.listdir(root_dir):
    number_of_images[dir] = len(os.listdir(os.path.join(root_dir,dir)))
    
number_of_images.items()

def createdir(p,split):
    if not os.path.exists("./"+p):
        os.mkdir("./"+p)
        for dir in os.listdir(root_dir):
            os.makedirs("./"+p+"/"+dir)
            for img in np.random.choice(a = os.listdir(os.path.join(root_dir,dir)),
                                        size = (math.floor(split*number_of_images[dir])-5),
                                               replace = False):
                O = os.path.join(root_dir,dir,img)
                D = os.path.join("./"+p,dir)
                shutil.copy(O,D)
                os.remove(O)
    else:
        print("Folder Exists")
            
createdir("train",0.7)
createdir("test",0.15)
createdir("val",0.15)

number_of_images={}
for dir in os.listdir(root_dir):
    number_of_images[dir] = len(os.listdir(os.path.join(root_dir,dir)))
    
number_of_images.items()

model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))

model.add(Conv2D(filters=36,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

def preprocess1(path):
    image_data = ImageDataGenerator(zoom_range=0.2,shear_range=0.2,rescale=1/255,horizontal_flip=True)
    image = image_data.flow_from_directory(directory=path,target_size=(224,224),batch_size=32,class_mode='binary')
    return image
def preprocess2(path):
    image_data = ImageDataGenerator(rescale=1/255)
    image = image_data.flow_from_directory(directory=path,target_size=(224,224),batch_size=32,class_mode='binary')
    return image

path1 = "C:/Users/Ulaganathan/train"
path2 = "C:/Users/Ulaganathan/val"
path3 = "C:/Users/Ulaganathan/test"

train_data = preprocess1(path1)
val_data = preprocess2(path2)
test_data = preprocess2(path3)

es = EarlyStopping(monitor='val_accuracy',min_delta=0.01,patience=6,verbose=1,mode='auto')
mc = ModelCheckpoint(monitor='val_accuracy',filepath='./bestmodel.h5',save_best_only=True,verbose=1,mode='auto')
cb = [es,mc]

model.fit(train_data,steps_per_epoch=8,epochs=30,verbose=1,validation_data=val_data,validation_steps=16,callbacks=cb)
model = load_model("C:/Users/Ulaganathan/bestmodel.h5")

acc = model.evaluate_generator(test_data)[1]
print(f"Your model's accuracy is {acc}")
