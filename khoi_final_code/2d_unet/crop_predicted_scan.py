"""
Cropping CT scan using xy predicted by yolov5
"""
import sys

from numpy import mat
from yaml import parse
sys.path.append("/wecare/home/khoi/thesis")

from utils.data_path import *
from utils.lesion_tools import unnormalize_lesion_props
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms.functional import crop
from torchvision.io import read_image
import torchvision.io
from tqdm import tqdm
from math import ceil
import argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-fold")
    parser.add_argument("--height")
    parser.add_argument("--width")
    args = parser.parse_args()

    PREDICTED = Path(RUNS_YOLOV5) / f"detect/holdout_{args.holdout_fold}"
    IMG_DIR = Path(AUGMENTED_SCANS)

    WIDTH = int(args.width)
    HEIGHT = int(args.height)
    for fold in tqdm(range(1,6)):
        """
        Read each label file to get a list of predicted bounding boxes for that slice. Then crop in the true mask 
        based on the predicted bounding boxes information 
        """
        FOLD_PREDICTED_LABELS = PREDICTED / f"fold_{fold}/labels"
        FOLD_SCANS = IMG_DIR / f"fold_{fold}"
        
        all_labels = list(FOLD_PREDICTED_LABELS.glob("*.txt")) 
        """
        In each slice, information for each lesion is on a line. We load the true mask correspond to that slice.
        We then take out information for each lesion on that slice, unnormalize it, then crop the true mask based on it.
        Finally we save the cropped image from mask into TRUE/fold_{fold}/crops 
        """
        for slice in all_labels:
            img = read_image(str(FOLD_SCANS / f"{slice.stem}.png"), mode = torchvision.io.ImageReadMode.GRAY)[0]

            with open(str(slice), "r") as f:
                for count, line in enumerate(f):
                    _, center_x, center_y, bbox_width, bbox_height = map(float, line.split(" "))
                    props_unnorm = unnormalize_lesion_props((center_x, center_y), (bbox_width, bbox_height), (768, 768)) 
                    top = ceil(props_unnorm[0][1] - (HEIGHT / 2))
                    left = ceil(props_unnorm[0][0] - (WIDTH / 2))
                    cropped_mask = crop(img, top = top, left = left, height= HEIGHT, width= WIDTH)
                    plt.imsave(PREDICTED / f"fold_{fold}/crops/lesion/" / f"{slice.stem}{count + 1 if count > 0 else ''}.jpg", cropped_mask, cmap = plt.gray())



