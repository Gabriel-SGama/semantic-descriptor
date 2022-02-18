import cv2
import numpy as np
from glob import glob

from numpy import zeros_like
 
cityscapes_pallete_float = np.array([[0.        , 0.        , 0.        ],
                                     [0.        , 0.        , 0.        ],
                                     [0.        , 0.        , 0.        ],
                                     [0.        , 0.        , 0.        ],
                                     [0.07843137, 0.07843137, 0.07843137],
                                     [0.43529412, 0.29019608, 0.        ],
                                     [0.31764706, 0.        , 0.31764706],
                                     [0.50196078, 0.25098039, 0.50196078],
                                     [0.95686275, 0.1372549 , 0.90980392],
                                     [0.98039216, 0.66666667, 0.62745098],
                                     [0.90196078, 0.58823529, 0.54901961],
                                     [0.2745098 , 0.2745098 , 0.2745098 ],
                                     [0.4       , 0.4       , 0.61176471],
                                     [0.74509804, 0.6       , 0.6       ],
                                     [0.70588235, 0.64705882, 0.70588235],
                                     [0.58823529, 0.39215686, 0.39215686],
                                     [0.58823529, 0.47058824, 0.35294118],
                                     [0.6       , 0.6       , 0.6       ],
                                     [0.6       , 0.6       , 0.6       ],
                                     [0.98039216, 0.66666667, 0.11764706],
                                     [0.8627451 , 0.8627451 , 0.        ],
                                     [0.41960784, 0.55686275, 0.1372549 ],
                                     [0.59607843, 0.98431373, 0.59607843],
                                     [0.2745098 , 0.50980392, 0.70588235],
                                     [0.8627451 , 0.07843137, 0.23529412],
                                     [1.        , 0.        , 0.        ],
                                     [0.        , 0.        , 0.55686275],
                                     [0.        , 0.        , 0.2745098 ],
                                     [0.        , 0.23529412, 0.39215686],
                                     [0.        , 0.        , 0.35294118],
                                     [0.        , 0.        , 0.43137255],
                                     [0.        , 0.31372549, 0.39215686],
                                     [0.        , 0.        , 0.90196078],
                                     [0.46666667, 0.04313725, 0.1254902 ],
                                     [0.        , 0.        , 0.55686275]])

img_array = []
size = (0,0)
# print(sorted(glob.glob("image/*")))
# filename_list = sorted(glob("image/*"))
# print(sorted(glob("image/*")))
# print(filename_list)
DIRECTORY = "carla_seq/11/"

images_path = sorted(glob(DIRECTORY+"image_2/*.png"))
labels_path = sorted(glob(DIRECTORY+"semantic/*.png"))

for imgp, labelp in zip(images_path, labels_path):
	img = cv2.imread(imgp)
	label = cv2.imread(labelp)

	cv2.imshow("frame",img)
	cv2.imshow("label",label)
	color = zeros_like(img)
	cv2.imwrite(labelp, label[:,:,2])
	
	# print(label)
	color[:,:,:] = cityscapes_pallete_float[label[:,:,2]]*255
	color[:,:,:] = color[:,:,::-1]
	# print(color)
	color = color.astype(np.uint8)
	# print(color.shape)
	# color = cv2.cvtColor(color, cv2.COLOR_RGB2RGB)
	# print(type(color))
	cv2.imshow("color",color)
	cv2.waitKey(1)
 
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()


# import cv2 
# import numpy as np
# from pyparsing import col
# cityscapes_pallete = np.array([[0, 0, 0],	
# [70, 70, 70],	
# [100, 40, 40],	
# [55, 90, 80],	
# [220, 20, 60],	
# [153, 153, 153],	
# [157, 234, 50],	
# [128, 64, 128],	
# [244, 35, 232],	
# [107, 142, 35],	
# [0, 0, 142],	
# [102, 102, 156],	
# [220, 220, 0],	
# [70, 130, 180],	
# [81, 0, 81],	 
# [150, 100, 100],	
# [230, 150, 140],	
# [180, 165, 180],	
# [250, 170, 30],	
# [110, 190, 160],	
# [170, 120, 50],	
# [45, 60, 150],	
# [0, 0, 0]])

# # Create a video capture object, in this case we are reading the video from a file
# vid_capture = cv2.VideoCapture('out_camera_kitti.avi')
# vid_seg = cv2.VideoCapture('out_seg_kitti.avi')

# if (vid_capture.isOpened() == False or vid_seg.isOpened() == False):
# 	print("Error opening the video file")

# else:
# 	# Get frame rate information
# 	fps = vid_capture.get(cv2.CAP_PROP_FPS)
# 	print('Frames per second : ', fps,'FPS')

# 	# Get frame count
# 	frame_count = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
# 	print('Frame count : ', frame_count)

# i=0
# while(vid_capture.isOpened() and vid_seg.isOpened()):
# 	# vid_capture.read() methods returns a tuple, first element is a bool 
# 	# and the second is frame
# 	ret, frame = vid_capture.read()
# 	ret_seg, frame_seg = vid_seg.read()
# 	color = np.zeros_like(frame_seg)
# 	if ret == True:

# 		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 		color = cityscapes_pallete_float[frame_seg[:,:,0]]*255
# 		color = color.astype(np.uint8)
# 		# color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

# 		# color = color[:,:,::-1]
# 		cv2.imshow('camera',frame)
# 		cv2.imshow('seg',frame_seg)
# 		cv2.imshow('color',color)
# 		cv2.waitKey(0)
# 		cv2.imwrite('image_2/' +str(i).zfill(6) + ".png", frame)
# 		cv2.imwrite('semantic/' +str(i).zfill(6) + ".png", frame_seg)

# 		# 20 is in milliseconds, try to increase the value, say 50 and observe
# 		key = cv2.waitKey(20)
		
# 		if key == ord('q'):
# 			break
# 		i+=1
# 	else:
# 		break

# # Release the video capture object
# vid_capture.release()
# vid_seg.release()
# cv2.destroyAllWindows()