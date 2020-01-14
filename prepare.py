from input_output import *
from edsr import *
from extract_frame import *
import imageio
import os

def load_model(scale = 4, batch_size = 16, num_resblocks = 16):

    os.makedirs("~/weights", exist_ok = True)
    train = DIV2K(scale, downgrade='bicubic', subset='train')
    train_ds = train.dataset(batch_size, random_transform=True)
    os.makedirs("~/weights", exist_ok = True)
    edsr_model = edsr(scale = scale, num_resblocks = num_resblocks)

    if(os.path.exists(os.path.join("~/weights", 'weights-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))):
        edsr_model.load_weights(os.path.join("~/weights", 'weights-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))
        edsr_model.summary()
        print("model loaded")
    else:
        optim_edsr = Adam(learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]))
        
        edsr_model.compile(optimizer=optim_edsr, loss='mean_absolute_error')
        edsr_model.fit(train_ds, epochs=300, steps_per_epoch=1000)

        edsr_model.save_weights(os.path.join("~/weights", 'weights-edsr-'+str(num_resblocks)+'-x'+str(scale)+'.h5'))

    return edsr_model
