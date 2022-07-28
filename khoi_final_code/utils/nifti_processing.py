#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue April 12 15:54 2022

@author: Khoi Quang Nguyen
@email: k.q.nguyen@tilburguniversity.edu
"""
import numpy as np
import nibabel as nib 
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label, regionprops
from .lesion_tools import get_lesion_props, normalize_lesions_props
from .raw_data_processing import Data_processing 

def nifti_to_png(path_to_nifti, path_to_png, file_names = [""], resolution = (768,768) ,load_all = False, mute = False): 
    """
    Parameters
    ----------
    path_to_nifti: path to the folder containing nifti files.
    path_to_png: path to the folder being used to save pngs.
    files_names: list of file_names to be processed. Must be a list even if there is only one file.
    load_all: If True then will load all files in path_to_nifti
    """
    all_files = os.listdir(path_to_nifti)
    files_list = [file for file in all_files if file.find(".nii")!=-1]
    nifti_loader = Data_processing()

    if load_all == True:
        files_to_load = files_list
    else:
        files_to_load = file_names
    for file_name in tqdm(files_to_load, disable = mute):
        scan_data, scan_header = nifti_loader.Loading_Nifti_data(path_to_nifti, file_name, Resize = True, Resolution = resolution, Mute = True)
        scan = scan_data[0]
        num_slices = scan.shape[2]
        for k in range(num_slices):
            plt.imsave(os.path.join(path_to_png, f"{file_name[:-4]}_{k:04}.png"), scan[:,:,k], cmap = plt.gray())
        
            
        

def mask_to_target(path_to_masks, path_to_targets, file_names = [""], resolution = (768, 768) ,load_all = False, mute = False):
    """
    Parameters
    ----------
    path_to_masks: path to masks in nifti format
    path_to_labels: path to where you want to save the target files.
    """
    all_masks = os.listdir(path_to_masks)
    masks_list = [file for file in all_masks if file.find(".nii")!=-1]
    nifti_loader = Data_processing()

    if load_all == True:
        masks_to_load = masks_list
    else:
        masks_to_load = file_names
    for mask_name in tqdm(masks_to_load, disable = mute):
        mask_data, mask_header = nifti_loader.Loading_Nifti_data(path_to_masks, mask_name, Resize=True, Resolution= resolution, Mute = True) 
        mask = mask_data[0]
        num_slices = mask.shape[2]
        for k in range(num_slices):
            lesion_props = get_lesion_props(mask[:,:,k])
            for lesion in range(len(lesion_props)):
                center = tuple(map(int, lesion_props[lesion].centroid))
                bbox_height, bbox_width = lesion_props[lesion].image.shape
                img_width, img_height, _ = mask.shape
                props_norm = normalize_lesions_props((center[1], center[0]), (bbox_width, bbox_height), (img_width, img_height))
                with open(os.path.join(path_to_targets, f"{mask_name[:-4]}_{k:04}.txt"), 'a') as f:
                    f.write(f"0 {props_norm[0][0]} {props_norm[0][1]} {props_norm[1][0]} {props_norm[1][1]}\n")


