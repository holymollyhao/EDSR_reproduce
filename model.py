from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
from data import DIV2K

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from data import DIV2K


#resdiual 기본 블록

def res_block(input, num_filters, resblock_scaling):

  #first convolution filter
  x = keras.layers.Conv2D(num_filters, 3, padding = 'same')(input)
  
  #relu
  x = keras.layers.Activation('relu')(x)

  #second convolution filter
  x = keras.layers.Conv2D(num_filters, 3, padding = 'same')(x)

  #residual scaling 
  x = x * 0.1 
  x = keras.layers.Add()[x, input]
  return x

def upscale_block(input, num_filters, scale):
  if scale == 2:
    x = keras.layers.Conv2D(num_flilters * (scale ** 2), 3, padding = 'same')(input)
    x = tf.nn.depth_to_space(x, scale)(x)

  elif scale == 3:
    x = keras.layers.Conv2D(num_flilters * (scale ** 2), 3, padding = 'same')(input)
    x = tf.nn.depth_to_space(x, scale)(x)

  elif scale ==4:
    x = keras.layers.Conv2D(num_flilters * (scale ** 2), 3, padding = 'same')(input)
    x = tf.nn.depth_to_space(x, scale)(x)

  return x


def edsr(scale = 2, num_filters = 64, num_resblocks = 16, resblock_scaling = None):

  input_image = keras.layers.Input(shape = (None, None, 3))

  x = keras.layers.Conv2D(num_filters, 3, padding = 'same')(input_image)
  x_orig = x

  for i in range(num_resblocks):
    x = res_block(x, num_filters, resblock_scaling)

  x = keras.layers.Conv2D(num_filters, 3, padding = 'same')(x)
  x = keras.layers.Add()[x, x_orig]

  x = upscale_block(x, num_filters, scale)
  
  return Model(input_image, x)

train = DIV2K(scale=4, downgrade='bicubic', subset='train')
train_ds = train.dataset(batch_size=16, random_transform=True)










