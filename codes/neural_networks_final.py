# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:50:55 2020

@author: Sam
"""

import tensorflow as tf
from matplotlib import pyplot as plt
#plt.style.use('dark_background')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3


import numpy as np
import os
import cv2


#-----------------------------   Choix des données  ----------------------------------


path_train = "chemin des données d'entrainement "
path_valid = "chemin des données de validation"
savedmodel = 'chemin et nom avec extension .h5 pour enregistrer le model"
paths = [path_train, path_valid]

CATEGORIES = ["label 1", "label 2", "..."]#nom des labels

IMG_SIZE = 90



#-----------------------------   extractions des labels  ----------------------------------

def create_data():
    
    train_data = []
    valid_data = []
    for path in paths :
        for img in os.listdir(path):
            img_class_num = img.split('_')[-1].split('.')[0]     
            try:
                img_array=cv2.imread(os.path.join(path, img))#, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))/255.0
                if path == path_train :
                    train_data.append([new_array, int(img_class_num)-1])
                elif path == path_valid :
                    valid_data.append([new_array, int(img_class_num)-1])
            except Exception:
                pass
    return (train_data, valid_data)

(valid_data, train_data) = create_data()


#-----------------------------  Données d'entrainements  ----------------------------------

train_features=[]
train_classes = []
for feature, classes in train_data :
    train_features.append(feature)
    train_classes.append(classes)

train_features=np.array(train_features).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
train_classes=np.array(train_classes)


#-----------------------------  Données de validation  ----------------------------------
valid_features=[]
valid_classes = []
for feature, classes in valid_data :
    valid_features.append(feature)
    valid_classes.append(classes)

valid_features=np.array(valid_features).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
valid_classes=np.array(valid_classes)



#---------------------------  Chargement et modifications modèle  ------------------------


Incep = InceptionV3(weights='imagenet', include_top=False, input_shape=(90, 90, 3))
#Incep.summary()

model =Sequential()
model.add(Incep)
model.add(Flatten())
model.add(Dense(6, activation= 'softmax'))


model.compile (optimizer= "adam", 
               loss='sparse_categorical_crossentropy', 
               metrics=['accuracy'])

history = model.fit(train_features, train_classes, epochs=10, verbose=1)

#---------------------------  Validation & enregistrement du modèle  ------------------------


test_loss, test_acc = model.evaluate(valid_features, valid_classes)
print("\n Val.loss = ", test_loss," - Val.acc = ", test_acc, "\n")

model.save(savedmodel)

predictions = model.predict(valid_features)
predictions_highest = [np.argmax(predictions[i]) for i in range(predictions.shape[0])]
#print(predictions)
print(np.unique(predictions_highest,return_counts=True))


#---------------------------  Création de la matrice de confusion  ------------------------


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(predictions_highest,valid_classes)
from museotoolbox import charts
tg = charts.PlotConfusionMatrix(cm.T)
tg.add_text()
tg.color_diagonal()
tg.add_x_labels(CATEGORIES)
tg.add_y_labels(CATEGORIES)
tg.add_f1()