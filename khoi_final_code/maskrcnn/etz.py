from asyncio import tasks
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, SemSegEvaluator 
from detectron2.data import MetadataCatalog, DatasetCatalog
from etz_loader import *
from functools import partial

train_imgs, val_imgs, test_imgs, data_root, nc, names = parse_config("scratch_holdout_5.yaml")

for img_dirs, d in zip([train_imgs, val_imgs, test_imgs], ["train", "val", "test"]):
    etz_func = partial(get_etz_dict, img_dirs, data_root)
    DatasetCatalog.register("etz_" + d, etz_func) 
    MetadataCatalog.get("etz_" + d).set(thing_classes=["lesion"])
    MetadataCatalog.get("etz_" + d).set(stuff_classes=["lesion"])
    MetadataCatalog.get("etz_" + d).set(ignore_label=0)

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="etz_train", filter_empty = True),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
        ],
        image_format="L",
        use_instance_mask=True,
        instance_mask_format = "bitmask",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="etz_val", filter_empty=False), # etz_val when training and etz_test when testing
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
        ],
        image_format="${...train.mapper.image_format}",
        instance_mask_format = "bitmask",
    ),
    num_workers=4,
)

dataloader.evaluator = L(DatasetEvaluators)(
    evaluators = [
    L(COCOEvaluator)(
        dataset_name="etz_val", # etz_val when training and etz_test when testing
        output_dir = "./holdout5_predict" # holdout1_predict when training and holdout1_test when testing
    ),
    #L(SemSegEvaluator)(
    #    dataset_name="etz_val",
    #    output_dir = "./etz_predict"
    #)
    ]
)
