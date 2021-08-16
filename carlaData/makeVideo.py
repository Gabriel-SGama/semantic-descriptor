import cv2
import numpy as np
from glob import glob
 
img_array = []
size = (0,0)
# print(sorted(glob.glob("image/*")))
# filename_list = sorted(glob("image/*"))
# print(sorted(glob("image/*")))
# print(filename_list)

for filename in sorted(glob("image/*")):
    print(filename)
    img = cv2.imread(filename)
    print(img.shape)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()