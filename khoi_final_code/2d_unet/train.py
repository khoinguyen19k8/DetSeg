from distutils.command.config import config
from numpy import histogram
import tensorboard
import tensorflow as tf
import tensorflow_addons as tfa
from metrics import focal_tversky_loss
import tensorflow.keras as keras
from tensorflow.data import Dataset
from tensorflow.io import read_file, decode_png
import sys

sys.path.append("/wecare/home/khoi/thesis")
import sys, os, yaml
from os.path import join
from utils.data_path import *
import metrics
from model_2Du_net import UNet
from Unet_utils import *
import argparse
import datetime

def train_model(
    model,
    train_ds,
    val_ds,
    loss_func,
    optimizer,
    metrics,
    callbacks,
    batch_size,
    epochs,
):
    model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)
    history = model.fit(
        x=train_ds,
        batch_size=batch_size,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


def do_train(args, config_dict):
    # Setting configurations

    holdout = config_dict["holdout"]
    IMG_DIR = config_dict["IMG_DIR"]
    MASK_DIR = config_dict["MASK_DIR"]
    OUTPUT_DIR = config_dict["OUTPUT_DIR"]
    BATCH_SIZE = config_dict["BATCH_SIZE"]
    EPOCHS = config_dict["EPOCHS"]
    INIT_LR = config_dict["INIT_LR"]
    MAX_LR = config_dict["MAX_LR"]
    PRETRAINED_WEIGHT = config_dict["PRETRAINED_WEIGHT"]

    if args.resume:
        pass
    else:
        # Load Tensorflow Dataset for training/val and test set
        train_ds, val_ds, test_ds = load_data_endToEnd(holdout, IMG_DIR, MASK_DIR)

        # Define the model and load pre-trained weights
        model = UNet(768, 768, 1, 16, 0.2)
        # model.load_weights(PRETRAINED_WEIGHT)

        # Setting training configurations
        steps_per_epoch = len(train_ds) // BATCH_SIZE
        lr_schedule = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=INIT_LR,
            maximal_learning_rate=MAX_LR,
            scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
            step_size=2 * steps_per_epoch,
        )
        loss_func = keras.losses.BinaryCrossentropy()
        # loss_func = focal_tversky_loss
        optimizer = keras.optimizers.Adam(lr_schedule)
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            join(OUTPUT_DIR, "best_model_endToEnd.h5"),
            monitor="val_dice_coef",
            mode="max",
            save_freq="epoch",
            save_best_only=True,
        )
        early_stopping_cb = keras.callbacks.EarlyStopping(
            patience=20, restore_best_weights=True
        )
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=f"logs/holdout_{holdout}_endToEnd/"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=1,
        )
        callbacks = [checkpoint_cb, early_stopping_cb, tensorboard_callback]

        history = train_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            loss_func=loss_func,
            optimizer=optimizer,
            metrics=[metrics.dice_coef],
            callbacks=callbacks,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )


def main():
    parser = argparse.ArgumentParser(description="Arguments for training U-net 2D")
    parser.add_argument("--config-file", help="path to config-file")
    parser.add_argument(
        "--resume",
        help="Whether to resume training or not, will check output_dir for model",
    )
    parser.add_argument("--device", help="GPU ID to train on")
    args = parser.parse_args()

    # Set GPU ID for training
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        # gpus = tf.config.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], True)
        

    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    do_train(args, config_dict)


if __name__ == "__main__":
    main()
