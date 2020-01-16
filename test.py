import imageio
import matplotlib.pyplot as plt


from input_output import *
from EDSR import *
from postprocess import *
from VIDEO_data import *
from preprocess import *




train = DIV2K(scale=4, downgrade='bicubic', subset='train')
train_ds = train.dataset(batch_size=16, random_transform=True)

train_vid = VIDEO(scale=4, downgrade='bicubic', subset='train')
train_vid_ds = train_vid.dataset(batch_size=16, random_transform=True)

edsr_model = load_edsr_model(scale = 4, batch_size = 64, num_resblocks = 20, num_filters = 32, resblock_scaling = 0.1)
nas_model = load_nas_model(scale = 4, batch_size = 64, num_resblocks = 20, num_filters = 32, resblock_scaling = 0.1)

image_srall(nas_model, "./dataset/testimages240", "./dataset/outputimages_sr_nas240")
imagetovid("./dataset/outputimages_sr_nas240", "./dataset", "output2.webm", 30)

'''image_srall(edsr_model, "./dataset/testimages240", "./dataset/outputimages_sr240")
imagetovid("./dataset/outputimages_sr240", "./dataset", "output.webm", 30)'''

'''vidtoimage("./dataset/240p_s0_d60.webm", "./dataset/outputimages240")
image_srall(edsr_model, "./dataset/outputimages240", "./dataset/outputimages_sr240")
imagetovid("./dataset/outputimages_sr240", "./dataset", "output.webm", 30)'''
