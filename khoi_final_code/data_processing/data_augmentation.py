import os
import sys
from pathlib import Path
from tkinter import image_names
from tqdm import tqdm
from utils import data, lesion_tools
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

    # Number of augmented instances for each lesion slice.

    k = int(sys.argv[1]) 

    print(f"Report on the total number of lesions before augmenting")
    os.system(f"cat {AUGMENTED_LABELS}/* | wc -l")

    print(f"Start augmenting data") 
    all_lesion_slices = [Path(f).stem for f in os.listdir(AUGMENTED_LABELS) if os.path.isfile(os.path.join(AUGMENTED_LABELS, f))]
    etz_data = data.ETZ(AUGMENTED_SCANS, AUGMENTED_MASKS, AUGMENTED_LABELS, transform=True, seed=None) 
    #etz_loader = DataLoader(etz_data)
    for slice in tqdm(all_lesion_slices):
        # slice is in the form 'CTP[0-9]{2,3}_00[1|2]_[0-9]{4}'
        i = 0
        while i < k:
            image_aug, mask_aug = etz_data[slice]
            image_aug, mask_aug = image_aug.squeeze(), mask_aug.squeeze() 
            lesion_props = lesion_tools.get_lesion_props(mask_aug, background=0) # 0 is black, 255 is white
            if len(lesion_props) > 0:
                for lesion in range(len(lesion_props)):
                    center = tuple(map(int, lesion_props[lesion].centroid))
                    bbox_height, bbox_width = lesion_props[lesion].image.shape
                    img_width, img_height = mask_aug.shape
                    props_norm = lesion_tools.normalize_lesions_props((center[1], center[0]), (bbox_width, bbox_height), (img_width, img_height))
                    # Write bounding box coordinates and centers of augmented images 
                    with open(os.path.join(AUGMENTED_LABELS, f"{slice}_aug{i:02}.txt"), 'a') as f:
                        f.write(f"0 {props_norm[0][0]} {props_norm[0][1]} {props_norm[1][0]} {props_norm[1][1]}\n")
                # Save the augmented image
                plt.imsave(os.path.join(AUGMENTED_SCANS, f"{slice}_aug{i:02}.png"),
                            image_aug,
                            cmap = plt.gray())
                # Save the augmented mask
                plt.imsave(os.path.join(AUGMENTED_MASKS, f"{slice}_aug{i:02}.png"),
                            mask_aug,
                            cmap = plt.gray())
                # We only increase i when the augmented image still has lesions. Else we generate an augmented image again.
                i += 1


    print(f"Report on the total number of lesions after augmenting")
    os.system(f"cat {AUGMENTED_LABELS}/* | wc -l")