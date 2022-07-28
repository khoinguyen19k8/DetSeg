import argparse
from logging.config import DEFAULT_LOGGING_CONFIG_PORT
import os
import yaml
import tensorflow as tf
import tensorflow.keras as keras
from Unet_utils import load_data, threshold_binarize
import sys
sys.path.append("/wecare/home/khoi/thesis")
import metrics
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

def do_eval(args, config_dict):
    holdout = config_dict["holdout"]
    IMG_DIR = config_dict["IMG_DIR"]
    MASK_DIR = config_dict["MASK_DIR"]
    THRESHOLD = float(args.threshold)
    MODEL_PATH = config_dict["MODEL_WEIGHT"]
    
    # Load test dataset
    test_ds = load_data(holdout, IMG_DIR, MASK_DIR, training=False).batch(1)
    # Load the model
    model = keras.models.load_model(MODEL_PATH, custom_objects = {"dice_coef" : metrics.dice_coef}, compile = False) 
    
    dice_accumulate = np.zeros(len(test_ds))
    metric_accumulate = {"dice": np.zeros(len(test_ds)), "precision": np.zeros(len(test_ds)), "recall": np.zeros(len(test_ds))}
    for i, (img, mask) in tqdm(enumerate(test_ds.as_numpy_iterator())):
        y_pred = model.predict(img)
        y_pred = threshold_binarize(y_pred, THRESHOLD)
        metric_accumulate["dice"][i] = metrics.dice_coef(mask.astype(np.float32), y_pred).numpy()
        metric_accumulate["precision"][i], metric_accumulate["recall"][i] = metrics.prec_recal(mask.astype(np.float32), y_pred)
    result = [np.mean(v) for v in metric_accumulate.values()]
    return result
def main():
    parser = argparse.ArgumentParser(description="Arguments for U-Net 2D evaluation")
    
    parser.add_argument("--config-file", help = "path to eval config file")
    # parser.add_argument("--model", help="path to model file")
    parser.add_argument("--threshold", help = "threshold value to binarize probability map")
    parser.add_argument("--device", help="GPU ID to do eval on")
    
    args = parser.parse_args()
    # Set GPU ID for training
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
    with open(args.config_file, "r") as f:
        global_config_dict = yaml.safe_load(f)

    set_size = len(global_config_dict["config_files"])
    aggregated_metrics = np.zeros((3, set_size)) # In the order of dice, precision, recall 
    for i, (config_file, weight_file) in enumerate(zip(global_config_dict["config_files"], global_config_dict["weight_files"])):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        config_dict["MODEL_WEIGHT"] = weight_file
        aggregated_metrics[:, i] = do_eval(args, config_dict)
        print(f"aggregated metrics for set {i + 1}: {aggregated_metrics[:, i]}")
    mean_dice, mean_prec, mean_recall = np.mean(aggregated_metrics, axis = -1) 
    std_dice, std_prec, std_recall = np.std(aggregated_metrics, axis = -1)

    print(f"Dice coef: {mean_dice}+-{std_dice}, Precision: {mean_prec}+-{std_prec}, Recall: {mean_recall}+-{std_recall}")

if __name__ == "__main__":
    main()