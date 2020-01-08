from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
from tensorflow import keras


import numpy as np
import matplotlib.pyplot as plt

#resdiual 기본 블록

def res_block(input_tensor, kernel_size, filters):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
    """
    filters1, filters2 = filters

    #first convolution filter, kernel_initializer는 optional
    x = keras.layers.Conv2D(filters1, kernel_size = (3, 3),
    				  padding ='same',
                      #kernel_initializer='he_normal',
                      #kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
                      )(input_tensor)
    #relu
    x = keras.layers.Activation('relu')(x)

    #second convolution filter
    x = keras.layers.Conv2D(filters2, kernel_size = (3, 3),
                      padding='same',
                      #kernel_initializer='he_normal',
                      #kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
                      )(x)

    #residual scaling 
    x = x * 0.1 

	x = keras.layers.Add()[x, input_tensor]

    return x