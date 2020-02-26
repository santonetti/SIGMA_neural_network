# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:50:55 2020

@author: Sam
"""

#import tensorflow as tf

import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
#from tensorflow.keras.applications.inception_v3 import InceptionV3

import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
import cv2

    
#######################   Importation des données
tiles = '30x30/'

model_path = 'modelpeupliers_v2.h5'

CATEGORIES = ["Peupliers", "Non peupliers"]
SIZE = 30
IMG_SIZE = False

def create_data(tr_or_vl='train',resize=False,undersampling=True):
    
    path = tiles+"{}/".format(tr_or_vl)
    
    # generae label 1
    poplars = [cv2.imread(i) for i in glob.glob(path+'*_1.png')]
    
    # generate label 2, undersampling
    nonpoplars = glob.glob(path+'*_2.png')
    if undersampling is True and tr_or_vl == 'train':
        np.random.seed(12)
        nonpoplars = np.random.permutation(nonpoplars)[:len(poplars)]

    tnonpoplars = []
    for i in nonpoplars:
        t = cv2.imread(i)
        if t.shape == (30,30,3):
            tnonpoplars.append(t)
    nonpoplars = tnonpoplars
            
    if resize:
        poplars = [cv2.resize(i,(resize,resize)) for i in poplars]
        nonpoplars = [cv2.resize(i,(resize,resize)) for i in nonpoplars]
            
    return np.concatenate((poplars,nonpoplars)),np.concatenate(([0]*len(poplars),[1]*len(nonpoplars)))

train_features,train_classes = create_data('train',resize=IMG_SIZE)

# =============================================================================
#  Model using InceptionV3   
# =============================================================================
# =============================================================================
# Incep = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# model = Sequential()
# model.add(Incep)
# model.add(Flatten())
# model.add(Dense(2, activation= 'sigmoid'))
# model.compile (optimizer= "adam", loss='sparse_categorical_crossentropy', metrics=['entropy'])
#  
# model.fit(train_features, train_classes, epochs=2, verbose=1,batch_size=64)
# 
# =============================================================================

# =============================================================================
# Learn model with homemade CNN
# =============================================================================
model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape = train_features.shape[1:], activation = 'sigmoid'))  #input_shape =valid_features.shape[1:],
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation = 'sigmoid'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(8, (3, 3), activation = 'sigmoid'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units = 64, activation = 'sigmoid'))
model.add(Flatten())
model.add(Dense(units = 2, activation = 'sigmoid'))

model.compile(optimizer= 'adam',
               loss='sparse_categorical_crossentropy',metrics=['accuracy'])
 
model.fit(train_features, train_classes, epochs=1000, verbose=1)
# model.summary()
model.save(model_path)

# =============================================================================
# Error on train
# =============================================================================

train_predict = model.predict_classes(train_features)
f1_train = f1_score(train_classes,train_predict)
cm_train = confusion_matrix(train_classes,train_predict)
print('Confusion matrix on train : \n'+str(cm_train))
print('f1 on train: \n'+str(f1_train))
np.savetxt('confusion_matrix_train.csv',cm_train)

# =============================================================================
# Error on valid
# =============================================================================

valid_features, valid_classes = create_data('valid',resize=IMG_SIZE)
test_loss, test_acc = model.evaluate(valid_features, valid_classes)
print("\n Val.loss = ", test_loss," - Val.acc = ", test_acc, "\n")

valid_pred = model.predict_classes(valid_features)
f1_valid = f1_score(valid_classes,valid_pred)
cm_valid = confusion_matrix(valid_classes,valid_pred)
print('Confusion matrix on valid : \n'+str(cm_valid))
print('f1 on valid: \n'+str(f1_valid))
np.savetxt('confusion_matrix_valid.csv',cm_valid)
