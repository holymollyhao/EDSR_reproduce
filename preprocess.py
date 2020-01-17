import imageio
import os
import tensorflow as tf

from input_output import *
from EDSR import *
from postprocess import *
from VIDEO_data import *


'''tb_hist = keras.callbacks.TensorBoard(log_dir=os.path.join("./graph", 'weights-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5')), histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train, Y_train, epochs=1000, batch_size=10, validation_data=(X_val, Y_val), callbacks=[tb_hist])
'''
class MODEL:
    def __init__(self, 
                 model_type = "nas_edsr",
                 scale = 4,
                 batch_size = 16,
                 num_resblocks = 16,
                 num_filters = 64,
                 resblock_scaling = None):
        self.scale = scale
        self.batch_size = batch_size
        self.num_resblocks = num_resblocks
        self.num_filters = num_filters
        self.resblock_scaling = resblock_scaling
        self.model = edsr(scale = scale, num_resblocks = num_resblocks, num_filters = num_filters, resblock_scaling = resblock_scaling)
    
    def train(self, data = "div2k" epochs = 300, steps_per_epoch = 1000, learning_rate =  1e-4):
        
        os.makedirs("~/weights", exist_ok = True)
        if (data == "vid"):
            train = DIV2K(scale, downgrade='bicubic', subset='train')
            train_ds = train.dataset(batch_size, random_transform=True)
        else:
            if(data != "div2k"):
                print("NO SUCH DATA TYPE, preceding in div2k dataset")
            train = VIDEO(scale, downgrade='bicubic', subset='train')
            train_ds = train.dataset(batch_size, random_transform=True)


        if(os.path.exists(os.path.join("~/weights", f'weigths-{self.model_type}-{self.num_resblocks}-x{self.scale}.h5'))):
            self.model.load_weights(os.path.join("~/weights", f'weigths-{self.model_type}-{self.num_resblocks}-x{self.scale}.h5))
            self.model.summary()
            print("model loaded")
        else :
            optim_edsr = optim_edsr = tf.optimizers.Adam(learning_rate= tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200000], values=[learning_rate, learning_rate / 2]))
            
            tb_hist = keras.callbacks.TensorBoard(log_dir=os.path.join("./graph", 'graphs-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'), update_freq=100000, write_graph=True, write_images=True)

            self.model.compile(optimizer=optim_edsr, loss='mean_absolute_error')
            self.model.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data = train_ds, validation_steps = 60, callbacks = [tb_hist])
            
            self.model.save_weights(os.path.join("~/weights", f'weigths-{self.model_type}-{self.num_resblocks}-x{self.scale}.h5'))





"""
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
        
        tb_hist = keras.callbacks.TensorBoard(log_dir=os.path.join("./graph", 'graphs-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'), update_freq=100000, write_graph=True, write_images=True)
        edsr_model.compile(optimizer=optim_edsr, loss='mean_absolute_error')
        edsr_model.fit(train_ds, epochs=300, steps_per_epoch=1000)
        
        '''dataset size / batch size 5280 * epochs = 200000'''
        edsr_model.save_weights(os.path.join("~/weights", 'weights-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))

    return edsr_model



def load_nas_model(scale = 4, batch_size = 16, num_resblocks = 16, num_filters = 64, resblock_scaling = None):


    os.makedirs("~/weights", exist_ok = True)
    train = VIDEO(scale, downgrade='bicubic', subset='train')
    train_ds = train.dataset(batch_size, random_transform=True)
    valid_ds = train.dataset(batch_size, random_transform=False)

    nas_edsr_model = edsr(scale = scale, num_resblocks = num_resblocks, num_filters = num_filters, resblock_scaling = resblock_scaling)
    """os.path.exists(os.path.join("~/weights", 'weights-nas-edsr-'+str(num_resblocks)+'-x'+str(scale)+'-valid.h5'))"""
    if(False):
        nas_edsr_model.load_weights(os.path.join("~/weights", 'weights-nas-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))
        nas_edsr_model.summary()
        print("model loaded")
    else:
        optim_edsr = tf.optimizers.Adam(learning_rate= tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]))
        
        tb_hist = keras.callbacks.TensorBoard(log_dir=os.path.join("./graph", 'graphs-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'), update_freq=100000, write_graph=True, write_images=True)
        nas_edsr_model.compile(optimizer=optim_edsr, loss='mean_absolute_error')
        nas_edsr_model.fit(train_ds, epochs=240, steps_per_epoch=850, validation_data = train_ds, validation_steps = 60, callbacks = [tb_hist])
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
        fine_tune_model.fit(train_ds, epochs=240, steps_per_epoch=850)
        fine_tune_model.save_weights(os.path.join("~/weights", 'weights-nas-edsr-'+str(num_resblocks)+'-x'+str(scale)+'_finetune.h5'))

    return fine_tune_model"""