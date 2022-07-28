import os
import pandas as pd
import sys
from utils.data_split import split_patterns

if __name__ == "__main__":
    ROOT = '/wecare/home/khoi/thesis'
    # CT scans and masks in raw nii format
    DATA_FOLDER = '/wecare/projects/Slicer_ready_data/Original Patient Data/Normalised_Recsaled_data'
    CT_SCANS = os.path.join(DATA_FOLDER,'Images')
    MASKS = os.path.join(DATA_FOLDER,'Labels_nii')
    # Dataset for the project
    AUGMENTED_FOLDER = os.path.join(ROOT, 'ct_processed_aug')
    AUGMENTED_SCANS = os.path.join(AUGMENTED_FOLDER, 'images') 
    AUGMENTED_MASKS = os.path.join(AUGMENTED_FOLDER, 'masks')
    AUGMENTED_LABELS = os.path.join(AUGMENTED_FOLDER, 'labels')
    HYPEPARAMETERS = os.path.join(ROOT, 'parameters')
    K_FOLDS = os.path.join(HYPEPARAMETERS, 'k_folds.csv')
    k = int(sys.argv[1])

    df = pd.read_csv(K_FOLDS)
    all_dirs = [AUGMENTED_SCANS, AUGMENTED_MASKS, AUGMENTED_LABELS]

    # Split slices in images, masks, and labels directories into corresponding k-fold.    
    for DIR in all_dirs:
        for fold in range(k):
            pattern_list = pd.Series(df.loc[:,f"fold_{fold+1}"]).dropna().tolist()
            split_patterns(DIR,
                            os.path.join(DIR, f"fold_{fold+1}"),
                            pattern_list)
    
    # Creating text files containing what files should be used for training in each fold session.
    # For ex: fold_1_train.txt contains all files in fold 2 - 5
    #for hold_out_fold in range(1,k+1):
    #    folds_for_train = [f"fold_{i}" for i in range(1, k+1) if i != hold_out_fold]
        

    # Creating text files containing what files should be used for validating in each fold session.
    # For ex: fold_1_val.txt contains all files in fold 1
