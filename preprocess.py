import imageio
import os
import tensorflow as tf

from input_output import *
from EDSR import *
from postprocess import *
from VIDEO_data import *


def load_edsr_model(scale = 4, batch_size = 16, num_resblocks = 16, num_filters = 64, resblock_scaling = None):


    os.makedirs("~/weights", exist_ok = True)
    train = DIV2K(scale, downgrade='bicubic', subset='train')
    train_ds = train.dataset(batch_size, random_transform=True)

    edsr_model = edsr(scale = scale, num_resblocks = num_resblocks, num_filters = num_filters, resblock_scaling = resblock_scaling)

    if(os.path.exists(os.path.join("~/weights", 'weights-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))):
        edsr_model.load_weights(os.path.join("~/weights", 'weights-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))
        edsr_model.summary()
        print("model loaded")
    else:
        optim_edsr = tf.optimizers.Adam(learning_rate= tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]))
        
        edsr_model.compile(optimizer=optim_edsr, loss='mean_absolute_error')
        edsr_model.fit(train_ds, epochs=300, steps_per_epoch=1000)
        '''dataset size / batch size 5280 * epochs = 200000'''
        edsr_model.save_weights(os.path.join("~/weights", 'weights-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))

    return edsr_model



def load_nas_model(scale = 4, batch_size = 16, num_resblocks = 16, num_filters = 64, resblock_scaling = None):


    os.makedirs("~/weights", exist_ok = True)
    train = VIDEO(scale, downgrade='bicubic', subset='train')
    train_ds = train.dataset(batch_size, random_transform=True)

    nas_edsr_model = edsr(scale = scale, num_resblocks = num_resblocks, num_filters = num_filters, resblock_scaling = resblock_scaling)

    if(os.path.exists(os.path.join("~/weights", 'weights-nas-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))):
        nas_edsr_model.load_weights(os.path.join("~/weights", 'weights-nas-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))
        nas_edsr_model.summary()
        print("model loaded")
    else:
        optim_edsr = tf.optimizers.Adam(learning_rate= tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]))
        
        nas_edsr_model.compile(optimizer=optim_edsr, loss='mean_absolute_error')
        nas_edsr_model.fit(train_ds, epochs=600, steps_per_epoch=330)

        nas_edsr_model.save_weights(os.path.join("~/weights", 'weights-nas-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))

    return nas_edsr_model




def edsr_to_nas_model(scale = 4, batch_size = 16, num_resblocks = 16, num_filters = 64, resblock_scaling = None):

    os.makedirs("~/weights", exist_ok = True)

    train = DIV2K(scale=4, downgrade='bicubic', subset='train')
    train_ds = train.dataset(batch_size=16, random_transform=True)

    train_vid = VIDEO(scale=4, downgrade='bicubic', subset='train')
    train_vid_ds = train_vid.dataset(batch_size=16, random_transform=True)

    fine_tune_model = edsr(scale = scale, num_resblocks = num_resblocks, num_filters = num_filters, resblock_scaling = resblock_scaling)

    if(os.path.exists(os.path.join("~/weights", 'weights-nas-edsr-'+str(num_resblocks)+'-x'+str(scale)+'_finetune.h5'))):
        fine_tune_model.load_weights(os.path.join("~/weights", 'weights-nas-edsr-'+str(num_resblocks)+'-x'+str(scale)+'_finetune.h5'))
        fine_tune_model.summary()
        print("model loaded")
    else:
        optim_edsr = tf.optimizers.Adam(learning_rate= tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200000], values=[1e-5, 5e-6]))
        fine_tune_model = load_edsr_model(scale = scale, batch_size = batch_size, num_resblocks = num_resblocks, num_filters = num_filters, resblock_scaling = resblock_scaling)
        fine_tune_model.compile(optimizer=optim_edsr, loss='mean_absolute_error')
        fine_tune_model.fit(train_ds, epochs=600, steps_per_epoch=330)
        fine_tune_model.save_weights(os.path.join("~/weights", 'weights-nas-edsr-'+str(num_resblocks)+'-x'+str(scale)+'_finetune.h5'))

    return fine_tune_model