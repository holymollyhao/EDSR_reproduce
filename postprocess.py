import cv2
import math
import os
import glob
import time

from input_output import *

def image_srall(model, input_imgpath, output_imgpath):
    count = 0 
    if( not os.path.exists(output_imgpath + '/1000.png')):
        os.makedirs(output_imgpath, exist_ok = True)
        img_array = []
        imgname = []
        cnt = 0
        for filename in glob.glob(input_imgpath + '/*.png'):
            imgname.append(int(filename.replace(input_imgpath + '/','').replace('.png','')))

        imgname.sort()
        for filename in imgname:
            start_time = time.time()
            lr = load_image(input_imgpath + '/' +str(filename)+'.png')
            sr = resolve(model, lr)
            print("%s" % (time.time() - start_time))
            plt.imsave(os.path.join(output_imgpath, '%d.png') % count, np.array(sr))
            count += 1


    

def imagetovid(imgpath, vidpath, vidname, fps):
    img_array = []
    imgname = []
    cnt = 0
    if(not os.path.exists(vidname)):
        for filename in glob.glob(imgpath + '/*.png'):
            imgname.append(int(filename.replace(imgpath + '/','').replace('.png','')))

        imgname.sort()

        for filename in imgname:
            print(filename)
            img = cv2.imread(imgpath + '/' +str(filename)+'.png')
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
            
            print("imagetovid img_arr #"+str(cnt))
            cnt += 1
        print(imgname)
        print(img_array[0])
        height,width, layers=img_array[0].shape
        out = cv2.VideoWriter(vidname,cv2.VideoWriter_fourcc(*'VP80'), fps, (width, height))
        cnt = 0

        for i in range(len(img_array)):
            print("imagetovid makingvid #"+str(cnt))
            out.write(img_array[i])
            cnt += 1
        
        out.release()
    else:
        print("video already exists")


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
    

    