# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:31:18 2020

@author: Sam
"""

import tensorflow as tf
from matplotlib import pyplot as plt
plt.style.use('dark_background')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D, Dropout,Flatten
from tensorflow.keras.utils import plot_model
#from keras_applications.inception_v3 import Inception_V3
import numpy as np
import os
import cv2


#-----------------------------   Choix des données  ----------------------------------
path_train = "/home/dynafor/Documents/NN/FinlandMaps/tiles/V2/30x30/train" #données d'entrainement
path_valid = "/home/dynafor/Documents/NN/FinlandMaps/tiles/V2/30x30/valid" #données de validation

paths = [path_train, path_valid]

CATEGORIES = ["Forest", "Mesic grassland", "Moist grassland", "Arable", "Pasture", "Peatland"]#labels

IMG_SIZE = 30 #taille des images de validation et entrainement 

#-----------------------------   Importation des données  ----------------------------------


def create_data():
    
    train_data = []
    valid_data = []
    for path in paths :
        for img in os.listdir(path):
            img_class_num = img.split('_')[-1].split('.')[0]     
            try:
                img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))/255.0
                if path == path_train :
                    train_data.append([new_array, int(img_class_num)-1])
                elif path == path_valid :
                    valid_data.append([new_array, int(img_class_num)-1])
            except Exception:
                pass
    return ( valid_data,train_data)

(valid_data, train_data) = create_data()


#-----------------------------  Réarrangement des données d'entrainement ----------------------------------


train_features=[]
train_classes = []
for feature, classes in train_data :
    train_features.append(feature)
    train_classes.append(classes)

train_features=np.array(train_features).reshape(-1, IMG_SIZE, IMG_SIZE,1)
train_classes=np.array(train_classes)

#-----------------------------   Réarrangement des données de validation  ----------------------------------

valid_features=[]
valid_classes = []
for feature, classes in valid_data :
    valid_features.append(feature)
    valid_classes.append(classes)

valid_features=np.array(valid_features).reshape(-1, IMG_SIZE, IMG_SIZE,1)
valid_classes=np.array(valid_classes)

#-----------------------------   Création du model  ----------------------------------
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape =valid_features.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.6))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(6, activation=tf.nn.softmax))


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#-----------------------------   Entrainement du model  ----------------------------------


model.fit(train_features, train_classes, epochs=150, verbose=1)

val_loss, val_acc = model.evaluate(valid_features, valid_classes)
print("\n Val.loss = ", val_loss," - Val.acc = ", val_acc, "\n")

model.save('nature.h5')
new_model = tf.keras.models.load_model('nature.h5')

predictions = new_model.predict(valid_features)
predictions_highest = [np.argmax(predictions[i]) for i in range(predictions.shape[0])]
print(predictions)
np.unique(predictions_highest,return_counts=True)
print(np.argmax(predictions[0]))


#-----------------------------   Affichage de resultats du model  ----------------------------------


for i in range (0,2):
    plt.imshow(valid_features[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title("Prediction : " + CATEGORIES[np.argmax(predictions[i])])
    plt.xlabel("Reality : " + CATEGORIES[valid_classes[i]])
    plt.show()
    
    
#-----------------------------   PNG de l'architecture du model   ----------------------------------

plot_model(new_model , to_file='/home/dynafor/Documents/NN/FinlandMaps/cnn_model.png', show_shapes=True, show_layer_names=True)

