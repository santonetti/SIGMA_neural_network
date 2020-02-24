#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Feb 21 10:47:18 2020
@author: dynafor
"""
import types
import tempfile
import os
import museotoolbox as mtb 
import gdal 
import numpy as np
import cv2 
import tensorflow as tf

# ============================================================================
#raster50cm = "/mnt/DATA/Cours/Sigma/Projets/1920/tld/peuplier/predictpeup/82_tif_50cm.tif"
#vector = "/mnt/DATA/Cours/Sigma/Projets/1920/tld/peuplier/predictpeup/masqueforet.gpkg"
#mask = '/mnt/DATA/Cours/Sigma/Projets/1920/tld/peuplier/predictpeup/masqueforet.tif'
#mtb.processing.image_mask_from_vector(vector,raster50cm,'/tmp/mask.tif',invert=False)
#
# vector= '/home/dynafor/Documents/NN/OrthoRVB/masqueforet.gpkg'
# 
# 
# image_mask_from_vector(vector,raster5m,'/tmp/mask.tif',invert=False)

# 
# =============================================================================


raster = 'raster à prédire '
mask = 'masque du raster '
out_predict = 'image en sortie'

rM = mtb.processing.RasterMath(raster,in_image_mask =mask,return_3d =True)

SIZE = 30
rescale_size = 90
rM.custom_block_size(SIZE,SIZE)

new_model = tf.keras.models.load_model('nom du model ')

def predict_dl(X,model,insize,rescalesize) : 
    try:
        # block de taille 30x30x3 et de tpye array
        if X.size == insize*insize*3 and isinstance(X,np.ndarray):
            X = X.data
            # si la donnée est à générer car de type memory
            if type(X)== memoryview:
                X = np.asarray(X)
                
            # on resize à 90x90x3 pour convenir au modèle
            X = cv2.resize(np.asarray(X),(rescalesize,rescalesize))/255.0
            # on ajoute une première dimension
            X = np.expand_dims(X,0)
            # on prédit
            X_pred = model.predict(X)
            X_pred = np.argmax(X_pred)+1 
            # on remplit le bloc de la prédiction
            X_res = np.full((insize,insize,1),int(X_pred),dtype=np.uint8) 
            
        else :   
            X_res = np.full(X.shape[:-1],0,dtype=np.uint8)
    except:
        # si je ne sais quelle erreur, zéro partout !
        X_res = np.full(X.shape[:-1],0,dtype=np.uint8)
    return X_res

rM.add_function(predict_dl, out_image=out_predict ,model=new_model,insize=SIZE,rescalesize=rescale_size)
rM.run('100M')