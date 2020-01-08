from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
from tensorflow import keras


import numpy as np
import matplotlib.pyplot as plt

#resdiual 기본 블록

def res_block(input_tensor, kernel_size, filters):
  
  # filters: tuple, 순서대로 필터 num
  # kernel_size: tuple, kernel size 입력
  # input_tensor: 텐서가 들어간다 


  filters1, filters2 = filters

  #first convolution filter, kernel_initializer는 optional
  x = keras.layers.Conv2D(filters1, kernel_size,
  				          padding ='same',
                    #kernel_initializer='he_normal',
                    #kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
                    )(input_tensor)
  #relu
  x = keras.layers.Activation('relu')(x)

  #second convolution filter
  x = keras.layers.Conv2D(filters2, kernel_size,
                    padding='same',
                    #kernel_initializer='he_normal',
                    #kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
                    )(x)

  #residual scaling 
  x = x * 0.1 

  x = keras.layers.Add()[x, input_tensor]

  return x

def edsr(input_tensor, scale, num_layers):

  kernel_size = (3, 3)

  #frist convolution
  x = keras.layers.Conv2D(256, kernel_size,
                    padding ='same',
                    #kernel_initializer='he_normal',
                    #kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
                    )(input_tensor)

  #save data for future addition
  x_pre = x 

  #resblocks
  for i in range(num_layers):
    x = res_block(x, kernel_size, (256, 256))
  
  x = keras.layers.Conv2D(256, kernel_size,
                    padding ='same',
                    #kernel_initializer='he_normal',
                    #kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
                    )(x)
  x = x + x_pre











