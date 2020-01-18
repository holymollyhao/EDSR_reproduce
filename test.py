import imageio
import matplotlib.pyplot as plt
import cv2

from input_output import *
from EDSR import *
from postprocess import *
from VIDEO_data import *
from preprocess import *

#loading models with class model
edsr_model = MODEL(scale = 4,
                 model_type = "edsr",
                 batch_size = 64,
                 num_resblocks = 16,
                 num_filters = 64,
                 resblock_scaling = 0.1).train(data = "div2k", epochs = 300, steps_per_epoch = 1000, learning_rate =  1e-4, validation_steps = 60)

nas_edsr_model = MODEL(scale = 4,
                 model_type = "nas-edsr",
                 batch_size = 64,
                 num_resblocks = 20,
                 num_filters = 32,
                 resblock_scaling = 0.1).train(data = "vid", epochs = 240, steps_per_epoch = 850, learning_rate =  1e-4, validation_steps = 60)

#using functions to output sr-ed images
image_srall(edsr_model, "./dataset/testimages240", "./dataset/outputimages_sr_final1")
image_srall(nas_edsr_model, "./dataset/testimages240", "./dataset/outputimages_sr_final2")

#using functions to create videos
imagetovid("./dataset/outputimages_sr_final1", "", "output_final1.webm", 30)
imagetovid("./dataset/outputimages_sr_final2", "", "output_final2.webm", 30)

#psnr, ssim results of models
print_output("edsr", "./dataset/images/VIDEO_train_HR", "./dataset/outputimages_sr_final1")
print_output("nas-edsr", "./dataset/images/VIDEO_train_HR", "./dataset/outputimages_sr_final2")