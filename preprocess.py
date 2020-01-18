import imageio
import os
import tensorflow as tf

from input_output import *
from EDSR import *
from postprocess import *
from VIDEO_data import *

class MODEL:
    def __init__(self, 
                 model_type = "nas-edsr",
                 scale = 4,
                 batch_size = 16,
                 num_resblocks = 16,
                 num_filters = 64,
                 resblock_scaling = None):
        self.scale = scale
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_resblocks = num_resblocks
        self.num_filters = num_filters
        self.resblock_scaling = resblock_scaling
        self.model = edsr(scale = scale, num_resblocks = num_resblocks, num_filters = num_filters, resblock_scaling = resblock_scaling)
        '''
        EDSR Model : 
            num_filters : 16, 
            resblock_scaling : 0.1
            
        NAS-EDSR Model : 
            num_filters : 20, 
            resblock_scaling : 0.1
        '''

    def train(self, data = "div2k", epochs = 300, steps_per_epoch = 1000, learning_rate =  1e-4, validation_steps = 60):
        
        os.makedirs("parameters/weights", exist_ok = True)
        if (data == "vid"):
            train = DIV2K(self.scale, downgrade='bicubic', subset='train')
            train_ds = train.dataset(self.batch_size, random_transform=True)
        else:
            if(data != "div2k"):
                print("NO SUCH DATA TYPE, preceding in div2k dataset")
            train = VIDEO(self.scale, downgrade='bicubic', subset='train')
            train_ds = train.dataset(self.batch_size, random_transform=True)

        print(os.path.join("parameters/weights", f'weights-{self.model_type}-{self.num_resblocks}-x{self.scale}.h5'))
        if(os.path.exists(os.path.join("parameters/weights", f'weights-{self.model_type}-{self.num_resblocks}-x{self.scale}.h5'))):
            self.model.load_weights(os.path.join("parameters/weights", f'weights-{self.model_type}-{self.num_resblocks}-x{self.scale}.h5'))
            self.model.summary()
            print("model loaded")
        else :
            optim_edsr = optim_edsr = tf.optimizers.Adam(learning_rate= tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200000], values=[learning_rate, learning_rate / 2]))
            
            tb_hist = keras.callbacks.TensorBoard(log_dir=os.path.join("parameters/graph", f'graphs-{self.model_type}-{self.num_resblocks}-x{self.scale}'), update_freq=100000, write_graph=True, write_images=True)

            self.model.compile(optimizer=optim_edsr, loss='mean_absolute_error')
            self.model.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data = train_ds, validation_steps = validation_steps, callbacks = [tb_hist])
            
            self.model.save_weights(os.path.join("parameters/weights", f'weights-{self.model_type}-{self.num_resblocks}-x{self.scale}.h5'))
        
        return self.model

        '''
        EDSR Model : 
            data : div2k, 
            epochs : 300,
            steps_per_epoch : 1000,
            learning_rate : 1e-4,
            validation_steps : 60
            
        NAS-EDSR Model : 
            data : vid, 
            epochs : 240,
            steps_per_epoch : 850,
            learning_rate : 1e-4,
            validation_steps : 60
        '''


