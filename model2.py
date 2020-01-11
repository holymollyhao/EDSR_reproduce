from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from data import DIV2K
import os


#resdiual 

def res_block(input, num_filters, resblock_scaling):

  #first convolution filter
  x = keras.layers.Conv2D(num_filters, 3, padding = 'same')(input)
  
  #relu
  x = keras.layers.Activation('relu')(x)

  #second convolution filter
  x = keras.layers.Conv2D(num_filters, 3, padding = 'same')(x)

  #residual scaling 
  if(resblock_scaling):
    x = keras.layers.Lambda(lambda t: t * resblock_scaling)(x)
  x = keras.layers.Add()([x, input])

  return x

def upscale_block(input, scale, num_filters):
  if scale == 2:
    x = keras.layers.Conv2D(num_filters * (scale ** 2), 3, padding = 'same')(input)
    x = keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)

  elif scale == 3:
    x = keras.layers.Conv2D(num_filters * (scale ** 2), 3, padding = 'same')(input)
    x = keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)

  elif scale ==4:
    x = keras.layers.Conv2D(num_filters * (scale ** 2), 3, padding = 'same')(input)
    x = keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)

  return x

def edsr(scale = 2, num_filters = 64, num_resblocks = 16, resblock_scaling = None):

  input_image = keras.layers.Input(shape = (None, None, 3))

  x = keras.layers.Lambda(lambda x: x / 255.0)(input_image)

  x = keras.layers.Conv2D(num_filters, 3, padding = 'same')(x)
  x_orig = x

  for i in range(num_resblocks):
    x = res_block(x, num_filters, resblock_scaling)

  x = keras.layers.Conv2D(num_filters, 3, padding = 'same')(x)
  x = keras.layers.Add()([x, x_orig])

  x = upscale_block(x, scale, num_filters)
  
  x = keras.layers.Conv2D(3, 3, padding='same')(x)

  x = keras.layers.Lambda(lambda x: x * 255.0)(x)
  return keras.models.Model(input_image, x,name="edsr")

train = DIV2K(scale=4, downgrade='bicubic', subset='train')
train_ds = train.dataset(batch_size=16, random_transform=True)
os.makedirs("~/weights", exist_ok = True)
edsr_model = edsr(scale=4, num_resblocks=16)
print(tf.test.is_gpu_available())

adam = keras.optimizers.Adam(learning_rate=0.001)

edsr_model.compile(optimizer=adam,
              loss='mean_absolute_error',
              )
edsr_model.fit(train_ds, epochs=300, steps_per_epoch=1000)

edsr_model.save_weights(os.path.join("~/weights", 'weights-edsr-16-x4.h5'))


