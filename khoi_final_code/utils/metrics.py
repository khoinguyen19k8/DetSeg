import tensorflow as tf

def dice_coef(y_true, y_pred, threshold = 0.5, smooth=1):
    # y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    print(intersection)
    return (2 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)