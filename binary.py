#####################test mask
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
masks = []
k = [0,1]
for i in range(2):
    mask = plt.imread('./valid/segment/map1_1004.png')*255
    print(mask.shape)
    print(np.unique(mask))
    mask = np.where(mask == k[i],255, 0)
    masks.append(mask)
    cv2.imwrite("a.png",mask)
    #plt.imshow(mask)
    #plt.show()