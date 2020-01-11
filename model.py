'''
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
  x = x * 0.1 
  x = keras.layers.Add()([x, input])

  return x

def upscale_block(input, num_filters, scale):
  if scale == 2:
    x = keras.layers.Conv2D(num_filters * (scale ** 2), 3, padding = 'same')(input)
    x = keras.layers.Lambda(tf.nn.depth_to_space(x, scale))(x)

  elif scale == 3:
    x = keras.layers.Conv2D(num_filters * (scale ** 2), 3, padding = 'same')(input)
    x = keras.layers.Lambda(tf.nn.depth_to_space(x, scale))(x)

  elif scale ==4:
    x = keras.layers.Conv2D(num_filters * (scale ** 2), 3, padding = 'same')(input)
    x = tf.nn.depth_to_space(x, scale)

  return x


def upscale_block(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        """Sub-pixel convolution."""
        x = keras.layers.Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return keras.layers.Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def edsr(scale = 2, num_filters = 64, num_resblocks = 16, resblock_scaling = None):

  input_image = keras.layers.Input(shape = (None, None, 3))

  x = keras.layers.Conv2D(num_filters, 3, padding = 'same')(input_image)
  x_orig = x

  for i in range(num_resblocks):
    x = res_block(x, num_filters, resblock_scaling)

  x = keras.layers.Conv2D(num_filters, 3, padding = 'same')(x)
  x = keras.layers.Add()([x, x_orig])

  x = upscale_block(x, num_filters, scale)
  
  x = keras.layers.Conv2D(3, 3, padding='same')(x)

  return keras.models.Model(input_image, x)
'''

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

from data import DIV2K
import os


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    """Creates an EDSR model."""
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    """Creates an EDSR residual block."""
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        """Sub-pixel convolution."""
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def normalize(x):
    return (x - DIV2K_RGB_MEAN) / 127.5


def denormalize(x):
    return x * 127.5 + DIV2K_RGB_MEAN


train = DIV2K(scale=4, downgrade='bicubic', subset='train')
train_ds = train.dataset(batch_size=16, random_transform=True)
os.makedirs("~/weights", exist_ok = True)
edsr_model = edsr(scale = 4, num_filters =64, num_resblocks =16, resblock_scaling = 0.1)
print(tf.test.is_gpu_available())

adam = keras.optimizers.Adam(learning_rate=0.001)

edsr_model.compile(optimizer=adam,
              loss='mean_absolute_error',
              )
edsr_model.fit(train_ds, epochs=300, steps_per_epoch=1000)

model_edsr.save_weights(os.path.join("~/weights", 'weights-edsr-16-x4.h5'))






