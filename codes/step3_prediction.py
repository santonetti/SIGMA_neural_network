#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:47:18 2020

@author: dynafor
"""
import museotoolbox as mtb 
import numpy as np
import cv2 
import tensorflow as tf

# ============================================================================
#
# raster50cm = "/mnt/DATA/Cours/Sigma/Projets/1920/tld/peuplier/predictpeup/82_tif_50cm.tif"
# vector = "/mnt/DATA/Cours/Sigma/Projets/1920/tld/peuplier/predictpeup/masqueforet.gpkg"
# mask = '/mnt/DATA/Cours/Sigma/Projets/1920/tld/peuplier/predictpeup/masqueforet.tif'
# mtb.processing.image_mask_from_vector(vector,raster50cm,'/tmp/mask.tif',invert=False)
#
# vector= '/home/dynafor/Documents/NN/OrthoRVB/masqueforet.gpkg'
# 
# image_mask_from_vector(vector,raster5m,'/tmp/mask.tif',invert=False)
# 
# =============================================================================

raster = '82_tif_50cm.tif'
mask = 'masqueforet.tif'

rM = mtb.processing.RasterMath(raster,in_image_mask =mask,return_3d =True)

input_size = 30
rescale_size = False
rM.custom_block_size(input_size,input_size)

model = tf.keras.models.load_model('modelpeupliers.h5')

def predict_dl(X,model,input_size,rescale_size,divide_by=255) : 
    # block de taille 30x30x3 et de tpye array
    if X.size == input_size*input_size*3 and np.ma.isMaskedArray(X):
            
        # on resize à 90x90x3 pour convenir au modèle
        if rescale_size:
            X = cv2.resize(np.asarray(X),(rescale_size,rescale_size))
        
        X = X/divide_by
        # on ajoute une première dimension
        X = np.expand_dims(X,0)
        # on prédit
        X_pred = model.predict(X)
        X_pred = np.argmax(X_pred)+1 # +1 car hélène a peur des zéros et elle a raison
        # on remplit le bloc de la prédiction
        X_res = np.full((input_size,input_size,1),int(X_pred),dtype=np.uint8) 
        
    else :   
        X_res = np.full(X.shape[:-1],0,dtype=np.uint8)
    
    return X_res

rM.add_function(predict_dl, out_image='peupliers_predict.tif',model=model ,input_size=input_size,rescale_size=rescale_size)
rM.run('100M')
