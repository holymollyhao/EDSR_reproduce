import cv2
import math
import os
import glob
from input_output import *

def vidtoimage(videopath, imgpath):
    vidcap = cv2.VideoCapture(videopath)
    count = 0
    print("dd")
    print(os.path.exists(imgpath))
    if( not os.path.exists(imgpath)):   
        os.makedirs(imgpath)
        while vidcap.isOpened():
            success, image = vidcap.read()
            if success:
                cv2.imwrite(os.path.join(imgpath, '%d.png') % count, image)
                print("vidtoimage #"+str(count))
                count += 1
            else:
                break
        cv2.destroyAllWindows()
        vidcap.release()

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
            print(filename)
            lr = load_image(input_imgpath + '/' +str(filename)+'.png')
            sr = resolve(model, lr)
            plt.imsave(os.path.join(output_imgpath, '%d.png') % count, np.array(sr))
            print("image_srall #" + str(count))
            count += 1


    

def imagetovid(imgpath, vidpath, vidname, fps):
    img_array = []
    imgname = []
    cnt = 0
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

'''
frameRate = cap.get(1) #frame rate
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
cap.release()
'''