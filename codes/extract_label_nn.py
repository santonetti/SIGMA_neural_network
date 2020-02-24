#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:07:29 2020

@author: dynafor
"""
import os
import museotoolbox as mtb 
import gdal 
import numpy as np
import cv2 

#-----------------------------   Choix des données  ----------------------------------

raster = 'données rasters traitées...' 
size = 30
labels = [1,2]
Class = 'classnn'
Vector ='données vecteurs traitées...''
group = 'polygon_id'
saved_vector = 'chemin enregistrement vecteur et nom fichier'

#-----------   Enregistrement des données de validation et entrainement  --------------

cv =mtb.cross_validation.RandomStratifiedKFold(random_state=12)
cvs = cv.save_to_vector(Vector ,Class,  group ,out_vector = saved_vector )

mtb.processing.rasterize(raster,cvs[0][0],Class,'./train_50cm.tif',gdt=gdal.GDT_Byte)
mtb.processing.rasterize(raster,cvs[0][1],Class,'./valid_50cm.tif',gdt=gdal.GDT_Byte)

rM=mtb.processing.RasterMath(raster,return_3d=True)

rM.add_image('./train_50cm.tif')
rM.add_image('./valid_50cm.tif')

#---------------------------   Création des dossiers  ----------------------------------
folder = "./tiles/"
out_folder = folder+'{0}x{0}/'.format(size)
out_folder_trvl = out_folder+'{}/'

for trvl in ['train','valid']:
    tmp_out = out_folder_trvl.format(trvl)
    if not os.path.exists(tmp_out):
        print(tmp_out)
        os.makedirs(tmp_out)
rM.custom_block_size(size,size)

#---------------------------   Verification des labels  ----------------------------------

def check (X): 
    
    if np.all(X[1] == 0) and np.all(X[2] == 0):
        yield False,False

    elif np.all(np.in1d(X[1],labels)):
        label = int(X[1][0][0])
        yield label,'train'
    
    elif np.all(np.in1d(X[2],labels)):
        label = int(X[2][0][0])
        yield label,'valid'

#----------------------   Enrgistrement des images avec labels  ----------------------------

from tqdm import trange

for i in trange(0,rM.n_blocks):
    block = rM.get_block(i)
    if block[0].shape[0] == size:
        for label,trvl in check(block):
#        np.save('/home/dynafor/Documents/NN/FinlandMaps/img/{}/block_{}_{}_{}_{}.npy'.format(trvl,i,size_w,size_h,label),block.data)
           cv2.imwrite(os.path.join(out_folder_trvl.format(trvl),'block_{}_{}.png'.format(i, label)),block[0].data)
    