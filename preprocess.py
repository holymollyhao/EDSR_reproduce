import imageio
import os
import tensorflow as tf

from input_output import *
from EDSR import *
from postprocess import *
from VIDEO_data import *

def print_output(model_type, validation_image_input,  validation_image_output): 

    psnr_tensor = tf.image.psnr(load_image(validation_image_input +'/0000.png'), load_image(validation_image_output+'/0.png')[...,:3],max_val=255)
    ssim_tensor = tf.image.ssim(tf.convert_to_tensor(load_image(validation_image_input+'/0000.png')), tf.convert_to_tensor(load_image(validation_image_output+'/0.png')[...,:3]),max_val=255)
    print('\n')
    print(f'{model_type} model output:')
    print("PSNR:", end='')
    print(psnr_tensor.numpy())
    print("SSIM:", end='')
    print(ssim_tensor.numpy())
    print('\n')


class CustomHistory(keras.callbacks.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.accuracy_psnr = [] 
        self.accuracy_ssim = []
        self.epoch = 0
        
    def on_epoch_end(self, batch, logs={}):

        psnr_tensor = tf.image.psnr(load_image("./dataset/images/VIDEO_train_HR" + '/0000.png'), resolve(self.model, load_image("./dataset/images/VIDEO_train_LR_bicubic" + '/0000x4.png')),max_val=255)
        ssim_tensor = tf.image.ssim(tf.convert_to_tensor(load_image("./dataset/images/VIDEO_train_HR"+'/0000.png')), tf.convert_to_tensor(resolve(self.model, load_image("./dataset/images/VIDEO_train_LR_bicubic" + '/0000x4.png'))),max_val=255)

        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.accuracy_psnr.append(psnr_tensor.numpy())
        self.accuracy_ssim.append(ssim_tensor.numpy())
        self.epoch += 1
        print(" PSNR Value :" + str(psnr_tensor.numpy()))
        


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

    def train(self, data = "div2k", epochs = 300, steps_per_epoch = 1000, learning_rate =  1e-4):
        
        os.makedirs("parameters/weights", exist_ok = True)
        if (data == "vid"):
            train = VIDEO(self.scale, downgrade='bicubic', subset='train')
            train_ds = train.dataset(self.batch_size, random_transform=True)
        else:
            if(data != "div2k"):
                print("NO SUCH DATA TYPE, preceding in div2k dataset")
            train = DIV2K(self.scale, downgrade='bicubic', subset='train')
            train_ds = train.dataset(self.batch_size, random_transform=True)

        valid = VIDEO(self.scale, downgrade='bicubic', subset = 'train')
        valid_ds =  valid.dataset(batch_size = 1, random_transform=False, repeat_count =1)

        print(os.path.join("parameters/weights", f'weights-{self.model_type}-{self.num_resblocks}-x{self.scale}.h5'))
        if(os.path.exists(os.path.join("parameters/weights", f'weights-{self.model_type}-{self.num_resblocks}-x{self.scale}1.h5'))):
            self.model.load_weights(os.path.join("parameters/weights", f'weights-{self.model_type}-{self.num_resblocks}-x{self.scale}.h5'))
            print("model loaded")
        else :
            optim_edsr = optim_edsr = tf.optimizers.Adam(learning_rate= tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200000], values=[learning_rate, learning_rate / 2]))
        
            tb_hist = keras.callbacks.TensorBoard(log_dir=os.path.join("parameters/graph", f'graphs-{self.model_type}-{self.num_resblocks}-x{self.scale}'), update_freq=100000, write_graph=True, write_images=True)
            cust_hist = CustomHistory()

            self.model.compile(optimizer=optim_edsr, loss='mean_absolute_error')
            self.model.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data = valid_ds, callbacks = [tb_hist, cust_hist])
            
            self.model.save_weights(os.path.join("parameters/weights", f'weights-{self.model_type}-{self.num_resblocks}-x{self.scale}.h5'))

            os.makedirs(f"./trainhistory/{self.model_type}", exist_ok = True)
            np.savetxt(f"./trainhistory/{self.model_type}" + "train_loss.txt", cust_hist.train_loss, fmt="%s")
            np.savetxt(f"./trainhistory/{self.model_type}" + "val_loss.txt", cust_hist.val_loss, fmt="%s")
            np.savetxt(f"./trainhistory/{self.model_type}" + "train_acc.txt", cust_hist.train_acc, fmt="%s")
            np.savetxt(f"./trainhistory/{self.model_type}" + "val_acc.txt", cust_hist.val_acc, fmt="%s")
            np.savetxt(f"./trainhistory/{self.model_type}" + "accuracy_psnr.txt", cust_hist.accuracy_psnr, fmt="%s")
            np.savetxt(f"./trainhistory/{self.model_type}" + "accuracy_ssim.txt", cust_hist.accuracy_ssim, fmt="%s")

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


