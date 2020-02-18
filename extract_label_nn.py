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


mapcv = '/home/dynafor/Documents/NN/FinlandMaps/example_entire_map.tif'
size = 30
labels = [1,2,3,4,5,6]



cv =mtb.cross_validation.RandomStratifiedKFold(random_state=12)
cvs = cv.save_to_vector('/home/dynafor/Documents/NN/FinlandMaps/trainingV2.gpkg','Class', 'polygon_id' ,out_vector = '/home/dynafor/Documents/NN/FinlandMaps/crossvalV2.gpkg')

mtb.processing.rasterize(mapcv,cvs[0][0],'Class','/home/dynafor/Documents/NN/FinlandMaps/trainV2.tif',gdt=gdal.GDT_Byte)
mtb.processing.rasterize(mapcv,cvs[0][1],'Class','/home/dynafor/Documents/NN/FinlandMaps/validV2.tif',gdt=gdal.GDT_Byte)

rM=mtb.processing.RasterMath(mapcv,return_3d=True)

rM.add_image('/home/dynafor/Documents/NN/FinlandMaps/trainV2.tif')
rM.add_image('/home/dynafor/Documents/NN/FinlandMaps/validV2.tif')


folder = "/home/dynafor/Documents/NN/FinlandMaps/tiles/V2/"
#folder ="/home/dynafor/Documents/NN/FinlandMaps/tiles/"
out_folder = folder+'{0}x{0}/'.format(size)
out_folder_trvl = out_folder+'{}/'

for trvl in ['train','valid']:
    tmp_out = out_folder_trvl.format(trvl)
    if not os.path.exists(tmp_out):
        print(tmp_out)
        os.makedirs(tmp_out)
rM.custom_block_size(size,size)

def check (X): 
    
    if np.all(X[1] == 0) and np.all(X[2] == 0):
        yield False,False

    elif np.all(np.in1d(X[1],labels)):
        label = int(X[1][0][0])
        yield label,'train'
    
    elif np.all(np.in1d(X[2],labels)):
        label = int(X[2][0][0])
        yield label,'valid'


from tqdm import trange

for i in trange(0,rM.n_blocks):
    block = rM.get_block(i)
    if block[0].shape[0] == size:
        for label,trvl in check(block):
#        np.save('/home/dynafor/Documents/NN/FinlandMaps/img/{}/block_{}_{}_{}_{}.npy'.format(trvl,i,size_w,size_h,label),block.data)
           cv2.imwrite(os.path.join(out_folder_trvl.format(trvl),'block_{}_{}.png'.format(i, label)),block[0].data)
    