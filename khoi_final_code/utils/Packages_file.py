#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 08:18:24 2021

@author: mleeuwen
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys 
import time
import nrrd
from tqdm import tqdm
from pathlib import Path
from pydicom import dcmread
from pydicom import read_file
import nibabel as nib
from sklearn.feature_extraction import image
from skimage.measure import label, regionprops
import random
import tensorflow as tf
import nrrd
from sklearn.metrics import jaccard_score, precision_score, recall_score
from scipy.spatial.distance import dice, directed_hausdorff
import shutil
from sklearn.model_selection import train_test_split
import xlsxwriter
from sklearn.metrics import f1_score
import dicom2nifti
from sklearn.model_selection import KFold
import pandas as pd
import cv2
import pickle 
import openpyxl
from skimage import color, io
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Dropout,Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import *
from scipy.ndimage import zoom

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

