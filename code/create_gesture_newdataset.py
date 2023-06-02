#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

background = None

ROI_top = 100
ROI_bottom = 300
ROI_right = 350
ROI_left = 550


# In[2]:


def setBackground(frame):
    
    background = frame
    
    return background


# In[3]:


def applyConnectedComponent(threshold_frame, connectivity):
    
    output = cv2.connectedComponentsWithStats(thresholded, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    area = []
    for i in range(1, numLabels):
       area.append( stats[i, cv2.CC_STAT_AREA].item()) 
    maxArea = max(area)
    compIndex = area.index(maxArea)+1

    #Mask Noisy Components
    mask = np.zeros(thresholded.shape, dtype="uint8")
    for i in range(0, numLabels):
       if(i!=compIndex):
         componentMask = (labels==i).astype("uint8")*255
         mask = cv2.bitwise_or(mask, componentMask)

    return mask


# In[4]:


num_frames = 0
num_imgs_taken = 1001
element = "0"
threshold = 20
background = None
thresholded_frame = None
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    num_frames +=1
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
    
    
    
    if num_frames <= 60:
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        if num_frames==59:
           background = setBackground(gray_frame)
    elif num_frames>60 and num_frames<=300:
        cv2.putText(frame_copy, "Adjust Hand Gesture in ROI for"+ str(element), (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
       
           
    else:
        if background is not None:
                    
            #Taking difference btween background and foreground
            difference = cv2.absdiff(gray_frame, background)
            _ , thresholded = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
            
            #Applying Connected component
            thresholded_frame = applyConnectedComponent(thresholded, 4)
            cv2.imshow("Masked frame", thresholded_frame)
            
            if num_imgs_taken <= 10000:
                 
                cv2.putText(frame_copy, "Creating Dataset for"+ str(element)+ "and frames"+ str(num_imgs_taken), (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                cv2.imwrite(r"C:\Users\pc\anaconda3\Sign Language Recognition Project\DATASET\TRAIN\9"+"\\" + str(num_imgs_taken) + '.jpg', thresholded_frame)
                #cv2.imwrite(r"C:\Users\pc\Documents\project pics"+"\\" + str(num_imgs_taken) + '.jpg', roi)
                
            else:
                cv2.putText(frame_copy, "Data is stored", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.destroyAllWindows()
            num_imgs_taken += 1
    
    cv2.imshow("Sign Detection", frame_copy)
            
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()


# In[ ]:




