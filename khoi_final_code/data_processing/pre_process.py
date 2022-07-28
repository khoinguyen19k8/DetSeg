import os
import sys
from tkinter import W
from utils.nifti_processing import nifti_to_png, mask_to_target

if __name__ == "__main__":
    ROOT = '/wecare/home/khoi/thesis'
    DATA_FOLDER = '/wecare/projects/Slicer_ready_data/Original Patient Data/Normalised_Recsaled_data'
    CT_SCANS = os.path.join(DATA_FOLDER,'Images')
    MASKS = os.path.join(DATA_FOLDER,'Labels_nii')
    PROCESSED_FOLDER = os.path.join(ROOT, 'ct_processed')
    PROCESSED_SCANS = os.path.join(PROCESSED_FOLDER, 'scans') 
    PROCESSED_MASKS = os.path.join(PROCESSED_FOLDER, 'label_masks')
    PROCESSED_TARGET = os.path.join(PROCESSED_FOLDER, 'label_targets')

#    print("----------START CONVERTING CT SCANS INTO PNG----------\n")
#    nifti_to_png(CT_SCANS, PROCESSED_SCANS, load_all=True)  
#    print("----------START CONVERTING CT MASKS INTO PNG----------\n")
#    nifti_to_png(MASKS, PROCESSED_MASKS, load_all=True)
    print("----------START WRITING LESION TARGET INTO TXT----------\n")
    mask_to_target(MASKS, PROCESSED_TARGET, load_all=True) 
    print("----------END OF PRE-PROCESSING----------\n")
    print("PRE-PROCESSING STATISTICS\n")
    print("Size of processed scans")
    os.system(f"du -h {PROCESSED_SCANS}")
    print("Size of processed masks")
    os.system(f"du -h {PROCESSED_MASKS}") 

