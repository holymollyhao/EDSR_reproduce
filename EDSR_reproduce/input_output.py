import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def resolve(model, lr_batch):

    lr_batch = tf.cast(lr_batch, tf.float32)

    if(tf.rank(lr_batch)==3):
        lr_batch = tf.expand_dims(lr_batch, axis=0)
        sr_batch = model.predict(lr_batch)
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        return sr_batch[0]
    
    sr_batch = model.predict(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.cast(sr_batch, tf.uint8)

    return sr_batch

def load_image(path):
    return np.array(Image.open(path))

def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    