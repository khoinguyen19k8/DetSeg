import os
import re
from utils import data_split
import sys
import os
import csv

if __name__ == "__main__":
    
    ROOT = '/wecare/home/khoi/thesis'
    # CT scans and masks in raw nii format
    DATA_FOLDER = '/wecare/projects/Slicer_ready_data/Original Patient Data/Normalised_Recsaled_data'
    CT_SCANS = os.path.join(DATA_FOLDER,'Images')
    MASKS = os.path.join(DATA_FOLDER,'Labels_nii')
    # Dataset for the project
    PROCESSED_FOLDER = os.path.join(ROOT, 'ct_processed')
    PROCESSED_SCANS = os.path.join(PROCESSED_FOLDER, 'scans') 
    PROCESSED_MASKS = os.path.join(PROCESSED_FOLDER, 'label_masks')
    PROCESSED_TARGET = os.path.join(PROCESSED_FOLDER, 'label_targets')

    test_ratio = int(sys.argv[1]) # Get a constant number of test instances. In this project it is 11.

    # Get the list of instances assigned to test set. Ex: ["CTP02_001_0000",...]
    _, test_instances = data_split.get_split_instances(CT_SCANS, test_ratio=test_ratio)
    print(f"Number of instances in test set: {len(test_instances)}\n")
    
    # Spliting the processed CT scans into train and test set.
    print(20*"-")
    print("Start spliting CT scans")
    ct_scans_pattern = r'CTP[0-9]{2,3}_00[1|2]_0000'
    data_split.split_data(PROCESSED_SCANS, ct_scans_pattern, test_instances)

    # Modify the list of instances so that they are suitable with mask files. Ex: "CTP02_001_0000" -> "CTP02_001"
    ct_masks_pattern = r'CTP[0-9]{2,3}_00[1|2]'
    test_masks_instances = list(map(lambda x: re.search(ct_masks_pattern, x).group(0), test_instances)) 
    
    # Spliting the processed CT masks into train and test set.
    print(20*"-")
    print("Start spliting CT masks")
    data_split.split_data(PROCESSED_MASKS, ct_masks_pattern, test_masks_instances)

    # Spliting the label targets into train and test set.
    print(20*"-")
    print("Start spliting label targets")
    label_targets_pattern = r'CTP[0-9]{2,3}_00[1|2]'
    data_split.split_data(PROCESSED_TARGET, label_targets_pattern, test_masks_instances)
    print("Finish splitting the dataset")

    with open('test_instances_id.csv', 'w') as f:
        csv_w = csv.writer(f)
        for instance in test_instances:
            csv_w.writerow(instance) 
