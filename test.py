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
train_vid_ds = train.dataset(batch_size=16, random_transform=True)

edsr_model = load_model(scale = 4, batch_size = 64, num_resblocks = 20, num_filters = 32, resblock_scaling = 0.1)

'''vidtoimage("./dataset/240p_s0_d60.webm", "./dataset/outputimages240")
image_srall(edsr_model, "./dataset/outputimages240", "./dataset/outputimages_sr240")
imagetovid("./dataset/outputimages_sr240", "./dataset", "output.webm", 30)'''
