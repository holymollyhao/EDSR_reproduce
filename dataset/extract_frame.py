import cv2
import math

videoFile = "960p_s0_d60.webm"
imagesFolder = "/"
vidcap = cv2.VideoCapture(videoFile)
print(vidcap.get(cv2.CAP_PROP_FPS))
count=0
while(vidcap.isOpened()):
    ret, image = vidcap.read()
 
    if(int(vidcap.get(1)) % 30 == 0):
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        cv2.imwrite("frame%d.jpg" % count, image)
        print('Saved frame%d.jpg' % count)
        count += 1
vidcap.release()



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
print ("Done!")