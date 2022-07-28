import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.math import logical_and
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, zero_one_loss

def basic_metrics(y_true, y_pred):
    tp = K.sum(logical_and(y_pred == 1.0, y_true == 1.0))
    tn = K.sum(logical_and(y_pred == 0.0, y_true == 0.0))
    fp = K.sum(logical_and(y_pred == 1.0, y_true == 0.0))
    fn = K.sum(logical_and(y_pred == 0.0, y_true == 1.0))
    return tp, tn, fp, fn

def dice_coef(y_true, y_pred, threshold = 0.5, smooth=1):
    # y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # print(intersection)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return (1 - dice_coef(y_true, y_pred))

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

def prec_recal(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    precision = precision_score(y_true_f, y_pred_f, zero_division=0)
    recall = recall_score(y_true_f, y_pred_f, zero_division=0)
    return precision, recall
