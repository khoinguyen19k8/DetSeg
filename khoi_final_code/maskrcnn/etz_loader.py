import sys
sys.path.append("/wecare/home/khoi/thesis")
from utils.data_path import *
from utils.mask_processing import create_instance_mask
from detectron2.structures import BoxMode
from tqdm import tqdm
import numpy as np
import pycocotools
import yaml
from pathlib import Path
from os.path import normpath
from skimage.measure import regionprops
from torchvision.io import read_image
import torchvision.io

def parse_config(yaml_config):
    """
    Parse a yaaml config file and return relative file paths for images ids.
    --------------------
    Parameters:
    --------------------
    yaml_config (list[string]): list
    --------------------
    Yaml config format:
    path: Root path directory of the dataset
    train: a list contains text files or path relative to path. Ex: ['images/fold_2_train.txt', 'images/fold_3_train.txt', 'images/fold_4_train.txt', 'images/fold_5_train.txt']
    val: a list with the same format as above.
    test: a list with the same format as above. For the purpose of this project it's in the format of ['images/fold_1']
    nc: number of classes, background not count.
    names: list of class names.
    """
    with open(yaml_config, "r") as f:
        yaml_dict = yaml.safe_load(f)
    
    train_imgs = []
    val_imgs = []
    test_imgs = []
    
    data_root = Path(yaml_dict["path"])
    
    nc = yaml_dict["nc"]
    class_names = yaml_dict["names"]
    
    for instances_path in yaml_dict["train"]:
        full_instances_path = Path(data_root) / instances_path # Ex: /wecare/home/khoi/thesis/ct_processed_aug/images/fold_2_train.txt
        
        with open(full_instances_path, "r") as f:
            train_imgs.extend(normpath(line.rstrip()) for line in f)
    
    for instances_path in yaml_dict["val"]:
        full_instances_path = Path(data_root) / instances_path # Ex: /wecare/home/khoi/thesis/ct_processed_aug/images/fold_2_train.txt
        
        with open(full_instances_path, "r") as f:
            val_imgs.extend(normpath(line.rstrip()) for line in f)
    
    for instances_path in yaml_dict["test"]:
        full_instances_path = Path(data_root) / instances_path #Ex: /wecare/home/khoi/thesis/ct_processed_aug/images/fold_1
        
        test_imgs.extend('/'.join(file.parts[-2:]) for file in full_instances_path.iterdir())
        
    return train_imgs, val_imgs, test_imgs, str(data_root), nc, class_names

def get_etz_dict(imgs, data_root):
    """
    Parameters:
    img_dir_list (list[string]): list of absolute file paths.
    --------------------
    Fields needed for ETZ dataset:
        - file_name: full path to the image file. Create the absolute path to the file name.
        - height, width: the shape of the image
        - image_id: a unique id that identifies this image. It is the name of the image, after steaming file path and file extension.
        - annotations (list[dict]): Each dict needs to contains the following keys
            - bbox (list[float]): List of 4 numbers representing the bbox
            - bbox_mode (int): BoxMode.XYWH_ABS
            - category_id (int): 0
            - segmentation (dict):
                - Read in a mask image, binarize it, then convert it into COCO mask using pycocotools.mask.encode(np.asarray(mask, order="F")).
                - Also need to set cfg.INPUT.MASK_FORMAT = bitmask
        
        - Annotation list is empty if there is no lesion (Note: Set DATALOADER.FILTER_EMPTY_ANNOTATIONS = False to include it when training)
    --------------------
    Returns:
    list[dict] with a dict for each instance.
    --------------------
    """
    
    dataset_dicts = []
    for img in tqdm(imgs):
        record = {}
        
        record["file_name"] = str(Path(data_root) / "images" / img)
        record["height"] = 768
        record["width"] = 768
        record["image_id"] = Path(img).with_suffix('').stem
        record["sem_seg_file_name"] = str(Path(data_root) / "masks" / img)
        
        # Take out fold_2/CTP22_001_0271_aug09.txt  /wecare/home/khoi/thesis/ct_processed_aug/images/fold_2/CTP22_001_0271_aug09.png
        mask_absolute_path = Path(data_root) / "masks" / Path(img) 
        label_absolute_path = Path(data_root) / "labels" / Path(img).with_suffix('.txt')
        
        annos = []
        
        mask = read_image(str(mask_absolute_path), mode = torchvision.io.ImageReadMode.GRAY)[0].numpy()
        mask_map = create_instance_mask(mask, 0)
        
        if mask_map == 0:
            record["category_id"] = 1
            record["annotations"] = []
        else:
            for instance_mask in mask_map:
                instance_dict = {}
                lesion_prop = regionprops(instance_mask)[0]
                yc, xc = lesion_prop.centroid
                bbox_height, bbox_width = lesion_prop.image.shape
                y0 = yc - (bbox_height / 2)
                x0 = xc - (bbox_width / 2)
                instance_dict["bbox"] = [x0, y0, bbox_width, bbox_height]
                instance_dict["bbox_mode"] = BoxMode.XYWH_ABS
                instance_dict["category_id"] = 0
                coco_mask = pycocotools.mask.encode(np.asarray(instance_mask, order="F", dtype = np.uint8))
                instance_dict["segmentation"] = coco_mask
                annos.append(instance_dict)
            record["annotations"] = annos
        dataset_dicts.append(record)
    
    return dataset_dicts