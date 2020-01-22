import imageio
import matplotlib.pyplot as plt
import cv2

from input_output import *
from EDSR import *
from postprocess import *
from VIDEO_data import *
from preprocess import *

#loading models with class model

edsr = MODEL(scale = 4,
                 model_type = "edsr",
                 batch_size = 64,
                 num_resblocks = 20,
                 num_filters = 32,
                 resblock_scaling = 0.1)
edsr_model = edsr.train(data = "div2k", epochs = 300, steps_per_epoch = 1000, learning_rate =  1e-4)

nas_edsr_model = MODEL(scale = 4,
                 model_type = "nas-edsr",
                 batch_size = 64,
                 num_resblocks = 20,
                 num_filters = 32,
                resblock_scaling = 0.1).train(data = "vid", epochs = 1400, steps_per_epoch = 150, learning_rate =  1e-4) 

#train nas_edsr_model first on div2k dataset, then on vid dataset
fine_tune = MODEL(scale = 4,
                 model_type = "fine-tune-edsr",
                 batch_size = 64,
                 num_resblocks = 20,
                 num_filters = 32,
                 resblock_scaling = 0.1)
fine_tune.model = edsr.train(data = "vid", epochs = 1400, steps_per_epoch = 150, learning_rate =  1e-4)
fine_tune_model = fine_tune.model

#using functions to output sr-ed images
image_srall(edsr_model, "./dataset/testimages240", "./dataset/outputimages_sr_edsr")
image_srall(nas_edsr_model, "./dataset/testimages240", "./dataset/outputimages_sr_nas_edsr")
image_srall(nas_edsr_model, "./dataset/testimages240", "./dataset/outputimages_sr_fine_tune")

#using functions to create videos
imagetovid("./dataset/outputimages_sr_edsr", "", "video_edsr.webm", 30)
imagetovid("./dataset/outputimages_sr_nas_edsr", "", "video_nas_edsr.webm", 30)
imagetovid("./dataset/outputimages_sr_fine_tune", "", "video_fine_tune.webm", 30)

#psnr, ssim results of models
print_output("edsr_model", "./dataset/images/VIDEO_train_HR", "./dataset/outputimages_sr_edsr")
print_output("nas_edsr_model", "./dataset/images/VIDEO_train_HR", "./dataset/outputimages_sr_nas_edsr")
print_output("fine_tune_model", "./dataset/images/VIDEO_train_HR", "./dataset/outputimages_sr_fine_tune")

