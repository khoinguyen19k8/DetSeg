from cgi import test
import tensorflow as tf
from tensorflow.io import read_file, decode_png
import os
from glob import glob
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split
from os.path import join
from tensorflow.math import equal


def process(mask_dir,  x, end_to_end = False):
    """
    Process the path of the original image. Take out the fold number and image ID. Then process it
    to create a proper path for 2D U-Net training, val or testing. Return a pair of image, one for
    cropped image and cropped mask.
    --------------------
    Parameters:
    
    mask_dir: Directory that contains all the cropped true masks. For example, /wecare/home/khoi/thesis/runs/yolov5/detect/holdout_1_true
    x: absolute path to the image
    """
    splits = tf.strings.split(x, sep = os.sep)
    if end_to_end == False:
        fold, img_id = splits[-4], tf.strings.split(splits[-1], sep = ".")[0]
        img_path = x
        mask_path = tf.strings.join([tf.constant(mask_dir, dtype = tf.string),fold, "crops/lesion",img_id + '.png'], separator = os.sep)
    else:
        fold, img_id = splits[-2], tf.strings.split(splits[-1], sep = ".")[0]
        img_path = x
        mask_path = tf.strings.join([tf.constant(mask_dir, dtype = tf.string),fold, img_id + '.png'], separator = os.sep)


    img = decode_png(read_file(img_path), channels = 1)
    mask = decode_png(read_file(mask_path), channels = 1)
    # img = tf.image.resize(img, [192, 192], method = "nearest")
    # mask = tf.image.resize(mask, [192, 192], method = "nearest")
    mask = tf.cast(equal(mask, 255), tf.float32)
    return img, mask


def configure_for_performance(ds, batch_size=64):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size, drop_remainder = True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def load_data(holdout, img_dir, mask_dir, training = True):

    test_ds = Dataset.list_files(join(img_dir, f"fold_{holdout}/crops/lesion/*"))
    test_ds = test_ds.map(lambda x: process(mask_dir, x))
    
    if training == False:
        return test_ds

    # Get cropped images in all fold directories but holdout dir as training+val set.
    train_val_list = glob(join(img_dir, f"fold_[!{holdout}]*/crops/lesion/*"))
    train_list, val_list = train_test_split(
        train_val_list, test_size=0.1, random_state=99
    )

    # Create tf.data.DataSet
    train_ds = Dataset.from_tensor_slices(train_list)
    val_ds = Dataset.from_tensor_slices(val_list)
    assert train_ds.cardinality().numpy() == len(train_list)  # Sanity check
    assert val_ds.cardinality().numpy() == len(val_list)

    
    # Run pre-processing transformations on each element of the dataset
    train_ds = train_ds.map(lambda x: process(mask_dir, x))
    val_ds = val_ds.map(lambda x: process(mask_dir, x))

    # Configurations for performance. e.g., prefetch, batching, buffer, etc
    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    # test_ds = configure_for_performance(test_ds)

    return train_ds, val_ds, test_ds

def load_data_endToEnd(holdout, img_dir, mask_dir, training = True):

    test_ds = Dataset.list_files(join(img_dir, f"fold_{holdout}/*"))
    test_ds = test_ds.map(lambda x: process(mask_dir, x, end_to_end = True))
    
    if training == False:
        return test_ds

    # Get cropped images in all fold directories but holdout dir as training+val set.
    train_val_list = glob(join(img_dir, f"fold_[!{holdout}]*/*"))
    train_list, val_list = train_test_split(
        train_val_list, test_size=0.1, random_state=99
    )

    # Create tf.data.DataSet
    train_ds = Dataset.from_tensor_slices(train_list)
    val_ds = Dataset.from_tensor_slices(val_list)
    assert train_ds.cardinality().numpy() == len(train_list)  # Sanity check
    assert val_ds.cardinality().numpy() == len(val_list)

    
    # Run pre-processing transformations on each element of the dataset
    train_ds = train_ds.map(lambda x: process(mask_dir, x, end_to_end = True))
    val_ds = val_ds.map(lambda x: process(mask_dir, x, end_to_end = True))

    # Configurations for performance. e.g., prefetch, batching, buffer, etc
    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    # test_ds = configure_for_performance(test_ds)

    return train_ds, val_ds, test_ds
def threshold_binarize(prob_map, threshold = 0.9):
    """
    Binarize a probability map based on a threshold
    """
    higher_threshold_indices = prob_map > threshold
    lower_threshold_indices = prob_map <= threshold
    prob_map[higher_threshold_indices] = 1
    prob_map[lower_threshold_indices] = 0
    return prob_map
