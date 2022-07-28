from ntpath import join
from os.path import join

ROOT = '/wecare/home/khoi/thesis'
# CT scans and masks in raw nii format
DATA_FOLDER = '/wecare/projects/Slicer_ready_data/Original Patient Data/'
CT_SCANS = join(DATA_FOLDER,'Images_nii')
MASKS = join(DATA_FOLDER,'Corrected_labels_nii')

# Processed dataset
PROCESSED_FOLDER = join(ROOT, 'ct_processed')
PROCESSED_SCANS = join(PROCESSED_FOLDER, 'images')
PROCESSED_MASKS = join(PROCESSED_FOLDER, 'masks')
PROCESSED_LABELS = join(PROCESSED_FOLDER, 'labels')

# Augmented dataset for the project
AUGMENTED_FOLDER = join(ROOT, 'ct_processed_aug')
AUGMENTED_SCANS = join(AUGMENTED_FOLDER, 'images') 
AUGMENTED_MASKS = join(AUGMENTED_FOLDER, 'masks')
AUGMENTED_LABELS = join(AUGMENTED_FOLDER, 'labels')
HYPEPARAMETERS = join(ROOT, 'parameters')

# RUNS

RUNS_YOLOV5 = join(ROOT, "runs", "yolov5")
YOLOV5_DETECT = join(RUNS_YOLOV5, "detect")