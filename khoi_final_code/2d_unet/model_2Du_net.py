
import tensorflow as tf

def UNet(img_height,img_width,img_channels,filter_dim,dropout):
    inputs = tf.keras.layers.Input((img_height,img_width,img_channels))
    rescale = tf.keras.layers.Rescaling(scale = 1./255)(inputs)
    c1 = tf.keras.layers.Conv2D(filter_dim, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(rescale)
    c1 = tf.keras.layers.Dropout(dropout)(c1)
    c1 = tf.keras.layers.Conv2D(filter_dim, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = tf.keras.layers.Conv2D(filter_dim*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(dropout)(c2)
    c2 = tf.keras.layers.Conv2D(filter_dim*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(filter_dim*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(dropout)(c3)
    c3 = tf.keras.layers.Conv2D(filter_dim*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = tf.keras.layers.Conv2D(filter_dim*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(dropout)(c4)
    c4 = tf.keras.layers.Conv2D(filter_dim*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    
    c5 = tf.keras.layers.Conv2D(filter_dim*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(dropout)(c5)
    c5 = tf.keras.layers.Conv2D(filter_dim*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    u6 = tf.keras.layers.Conv2DTranspose(filter_dim*8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(filter_dim*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(dropout)(c6)
    c6 = tf.keras.layers.Conv2D(filter_dim*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(filter_dim*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(filter_dim*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(dropout)(c7)
    c7 = tf.keras.layers.Conv2D(filter_dim*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(filter_dim*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(filter_dim*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(dropout)(c8)
    c8 = tf.keras.layers.Conv2D(filter_dim*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = tf.keras.layers.Conv2DTranspose(filter_dim, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(filter_dim, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(dropout)(c9)
    c9 = tf.keras.layers.Conv2D(filter_dim, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model