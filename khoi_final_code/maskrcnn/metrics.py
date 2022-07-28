from cmath import e
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import DatasetCatalog
from sklearn.metrics import precision_score, recall_score
import pycocotools
import numpy as np

class SemSegEvaluator(DatasetEvaluator):
  """
  Calculate Dice coefficient, Precision, and Recall. Calculate dice coef, precision, and recall for each image.
  Then aggregate each metric mean and standard deviation.
  """
  def __init__(self, dataset_name):
    self.dataset = DatasetCatalog.get(dataset_name)
  def search(self, dataset, image_id):
    return list(filter(lambda image: image["file_name"] == image_id, dataset))[0]
  def dice_coef(self, y_true, y_pred, smooth = 1):
    """
    Calculate dice coef between 2 matrices.
    """
    intersection = np.sum(np.dot(y_true, y_pred))
    return (2 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
  def reset(self):
    self.dice_accumulate = []
    self.precision_accumulate = []
    self.recall_accumulate = []
  def process(self, inputs, outputs):
    for input, output in zip(inputs, outputs):
        img_size = output["instances"].image_size
        img_path = input["file_name"]
        image = self.search(self.dataset, img_path) # image dict corresponds to this image id
        # Create a semantic segmentation mask for true mask
        true_mask = np.zeros((img_size), dtype=np.uint8)
        if len(image["annotations"]) == 0:
            pass
        else:
            annos = image["annotations"] # This is a list of dict, each dict corresponds to one instance.
            for anno in annos:
                coco_instance_mask = anno["segmentation"]
                instance_mask = pycocotools.mask.decode(coco_instance_mask)
                true_mask = np.logical_or(true_mask, instance_mask)
            true_mask = true_mask.astype(np.uint8)
        # Create a semantic segmentation mask for predicted mask
        pred_mask = np.zeros((img_size))
        all_pred_masks = output["instances"].pred_masks
        for instance_mask in all_pred_masks:
            pred_mask = np.logical_or(pred_mask, instance_mask.cpu()).numpy()
        pred_mask = pred_mask.astype(np.uint8)
        # Calculate metrics and put into accumulate list
        true_mask = true_mask.flatten()
        pred_mask = pred_mask.flatten()
        self.dice_accumulate.append(self.dice_coef(true_mask, pred_mask))
        self.precision_accumulate.append(precision_score(true_mask, pred_mask, zero_division=0))
        self.recall_accumulate.append(recall_score(true_mask, pred_mask, zero_division=0))
  def evaluate(self):
    result = {"dice": {"mean": np.mean(self.dice_accumulate), "std": np.std(self.dice_accumulate) },
              "precision": {"mean": np.mean(self.precision_accumulate), "std": np.std(self.precision_accumulate)},
              "recall": {"mean": np.mean(self.recall_accumulate), "std": np.std(self.recall_accumulate)}}
    return result

class ObjectDetection(DatasetEvaluator):
    """
    Calculate precision and recall of bbox detection
    """
    def __init__(self, dataset_name):
        self.dataset = DatasetCatalog.get(dataset_name)
    def reset(self):
        self.precision_accumulate = []
        self.recall_accumulate = []
    def process(self):
        for input, output in zip(inputs, outputs):
            img_size = output["instances"].image_size
            img_path = input["file_name"]
            image = self.search(self.dataset, img_path) # image dict corresponds to this image id
            # Create a semantic segmentation mask for true mask
            true_mask = np.zeros((img_size), dtype=np.uint8)
            if len(image["annotations"]) == 0:
                pass
            else:
                annos = image["annotations"] # This is a list of dict, each dict corresponds to one instance.
                for anno in annos:
                    coco_instance_mask = anno["segmentation"]
                    instance_mask = pycocotools.mask.decode(coco_instance_mask)
                    true_mask = np.logical_or(true_mask, instance_mask)
                true_mask = true_mask.astype(np.uint8)
            # Create a semantic segmentation mask for predicted mask
            pred_mask = np.zeros((img_size))
            all_pred_masks = output["instances"].pred_masks
            for instance_mask in all_pred_masks:
                pred_mask = np.logical_or(pred_mask, instance_mask.cpu()).numpy()
            pred_mask = pred_mask.astype(np.uint8)
            # Calculate metrics and put into accumulate list
            true_mask = true_mask.flatten()
            pred_mask = pred_mask.flatten()
            self.dice_accumulate.append(self.dice_coef(true_mask, pred_mask))
            self.precision_accumulate.append(precision_score(true_mask, pred_mask, zero_division=0))
            self.recall_accumulate.append(recall_score(true_mask, pred_mask, zero_division=0))
    def evaluate(self):
        pass














