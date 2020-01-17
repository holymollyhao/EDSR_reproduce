import imageio
import matplotlib.pyplot as plt
import cv2

from input_output import *
from EDSR import *
from postprocess import *
from VIDEO_data import *
from preprocess import *




train = DIV2K(scale=4, downgrade='bicubic', subset='train')
train_ds = train.dataset(batch_size=64, random_transform=True)

train_vid = VIDEO(scale=4, downgrade='bicubic', subset='train')
train_vid_ds = train_vid.dataset(batch_size=64, random_transform=True)

nas_model = load_nas_model(scale = 4, batch_size = 64, num_resblocks = 20, num_filters = 32, resblock_scaling = 0.1)


'''
edsr_model = load_edsr_model(scale = 4, batch_size = 64, num_resblocks = 20, num_filters = 32, resblock_scaling = 0.1)
nas_model = load_nas_model(scale = 4, batch_size = 64, num_resblocks = 20, num_filters = 32, resblock_scaling = 0.1)
train_again_model = edsr_to_nas_model(scale = 4, batch_size = 64, num_resblocks = 20, num_filters = 32, resblock_scaling = 0.1)

print('\n')
print("EDSR model output:")
print("PSNR:")
print(tf.image.psnr(load_image('./dataset/images/VIDEO_train_HR/0000.png'), load_image('./dataset/outputimages_sr240/0.png')[...,:3],max_val=255))
print("SSIM:")
print(tf.image.ssim(tf.convert_to_tensor(load_image('./dataset/images/VIDEO_train_HR/0000.png')), tf.convert_to_tensor(load_image('./dataset/outputimages_sr240/0.png')[...,:3]),max_val=255))
print('\n')

print("NAS model output:")
print("PSNR:")
print(tf.image.psnr(load_image('./dataset/images/VIDEO_train_HR/0000.png'), load_image('./dataset/outputimages_sr_nas240/0.png')[...,:3],max_val=255))
print("SSIM:")
print(tf.image.ssim(tf.convert_to_tensor(load_image('./dataset/images/VIDEO_train_HR/0000.png')), tf.convert_to_tensor(load_image('./dataset/outputimages_sr_nas240/0.png')[...,:3]),max_val=255))
print('\n')

print("Fine-Tune model output:")
print("PSNR:")
print(tf.image.psnr(load_image('./dataset/images/VIDEO_train_HR/0000.png'), load_image('./dataset/outputimages_sr_finetune240/0.png')[...,:3],max_val=255))
print("SSIM:")
print(tf.image.ssim(tf.convert_to_tensor(load_image('./dataset/images/VIDEO_train_HR/0000.png')), tf.convert_to_tensor(load_image('./dataset/outputimages_sr_finetune240/0.png')[...,:3]),max_val=255))
print('\n')
'''