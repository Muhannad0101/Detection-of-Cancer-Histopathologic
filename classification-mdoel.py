import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import visualkeras
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import random
from random import randint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop,Adam,SGD,Adadelta
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize


df=pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')

Labels = df.label.values
def get_indexes(label,list_n):
    for x in range(len(Labels)):
        if Labels[x]==label:
            list_n.append(x)
    return list_n

def get_classlabel(class_code):
    labels = {0:'Non-cancerous', 1:'Cancerous'}
    
    return labels[class_code]

no_cancer=[]
no_cancer=get_indexes(0,no_cancer)
cancer=[]
cancer=get_indexes(1,cancer)

SAMPLE_SIZE=89000

df_0 = df[df['label'] == 0].sample(SAMPLE_SIZE, random_state = 42)
df_1 = df[df['label'] == 1].sample(SAMPLE_SIZE, random_state = 42)
df_data = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
df_data = shuffle(df_data)

y = df_data['label']
df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=42, stratify=y)

base_dir = 'base_dir'
os.mkdir(base_dir)


train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)


val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

non_cancerous_tissue = os.path.join(train_dir, 'non_cancerous_tissue')
os.mkdir(non_cancerous_tissue)
cancerous_tissue = os.path.join(train_dir, 'cancerous_tissue')
os.mkdir(cancerous_tissue)

non_cancerous_tissue = os.path.join(val_dir, 'non_cancerous_tissue')
os.mkdir(non_cancerous_tissue)
cancerous_tissue = os.path.join(val_dir, 'cancerous_tissue')
os.mkdir(cancerous_tissue)
df_data.set_index('id', inplace=True)

train_list = list(df_train['id'])
val_list = list(df_val['id'])

for image in train_list:
    fname = image + '.tif'
    target = df_data.loc[image,'label']
    
    if target == 0:
        label = 'non_cancerous_tissue'
    if target == 1:
        label = 'cancerous_tissue'
    
    src = os.path.join('../input/histopathologic-cancer-detection/train', fname)
    dst = os.path.join(train_dir, label, fname)
    shutil.copyfile(src, dst)


for image in val_list:
    fname = image + '.tif'
    target = df_data.loc[image,'label']
    
    if target == 0:
        label = 'non_cancerous_tissue'
    if target == 1:
        label = 'cancerous_tissue'
    
    src = os.path.join('../input/histopathologic-cancer-detection/train', fname)
    dst = os.path.join(val_dir, label, fname)
    shutil.copyfile(src, dst)

train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'
test_path = '../input/test'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32
val_batch_size = 32

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

IMAGE_SIZE=96
datagen = ImageDataGenerator(rescale=1.0/255,
                             featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=10,
                             zoom_range = 0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=True)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='binary',
                                        shuffle=True)

val_gen = datagen.flow_from_directory(valid_path,
                                      target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                      batch_size=val_batch_size,
                                      class_mode='binary',
                                      shuffle=True)

# Note: shuffle=False causes the test dataset to not be shuffled
val2_gen = datagen.flow_from_directory(valid_path,
                                       target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                       batch_size=1,
                                       class_mode='binary',
                                       shuffle=False)

#Will stop the training once it reaches 99% validation accuracy:
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True
        
callbacks = MyCallback()

#Will reduce the learning rate is validation accuracy didn't improve in one epoch:
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                 patience=1, 
                                 verbose=1, 
                                 factor=0.5, 
                                 min_lr=0.000001)

#Will save the very best model according to validation accuracy:
model_dir = 'CNN_model_histo.h5'
checkpoint = ModelCheckpoint(model_dir, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
optimizer = Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)

model=Sequential()
model.add(Conv2D(32,(3,3),strides=1,padding='Same',activation='relu',input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3), strides=1,padding= 'Same', activation='relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3), strides=1,padding= 'Same', activation='relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3), strides=1,padding= 'Same', activation='relu'))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])

history = model.fit_generator(train_gen, validation_data=val_gen,
                              epochs=20, verbose=1,
                              callbacks=[callbacks, lr_reduction, checkpoint])

model_saved = load_model('./CNN_model_histo.h5')
model_saved.evaluate_generator(val2_gen, steps=len(df_val), verbose=1)

# create test_dir
test_dir = 'test_set_dir'
os.mkdir(test_dir)
    
# create test_images inside test_dir
test_images = os.path.join(test_dir, 'test_images')
os.mkdir(test_images)
test_list = os.listdir('../input/histopathologic-cancer-detection/test')

for image in test_list:
    fname = image
    src = os.path.join('../input/histopathologic-cancer-detection/test', fname)
    dst = os.path.join(test_images, fname)
    shutil.copyfile(src, dst)

test_path = 'test_set_dir'
test_gen = datagen.flow_from_directory(test_path,
                                       target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                       batch_size=1,
                                       class_mode='categorical',
                                       shuffle=False)

test_predictions = model_saved.predict_generator(test_gen, 
                                                 steps=len(os.listdir('test_set_dir/test_images')), 
                                                 verbose=1)

non_cancerous = 1 - test_predictions
submission=pd.DataFrame(non_cancerous, columns=['label'])
submission['id']=test_gen.filenames
submission['id']=submission['id'].str.split('/', 
                                            n=1, 
                                            expand=True)[1].str.split('.', 
                                                                      n=1, 
                                                                      expand=True)[0] 
submission.set_index('id', inplace=True)
submission.to_csv('submission.csv')