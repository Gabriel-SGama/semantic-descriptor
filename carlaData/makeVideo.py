# import cv2
# import numpy as np
# from glob import glob
 
# img_array = []
# size = (0,0)
# # print(sorted(glob.glob("image/*")))
# # filename_list = sorted(glob("image/*"))
# # print(sorted(glob("image/*")))
# # print(filename_list)

# for filename in sorted(glob("image/*")):
#     print(filename)
#     img = cv2.imread(filename)
#     print(img.shape)
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)
 
 
# out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()


import cv2 

# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture('output.avi')

if (vid_capture.isOpened() == False):
	print("Error opening the video file")
# Read fps and frame count
else:
	# Get frame rate information
	# You can replace 5 with CAP_PROP_FPS as well, they are enumerations
	fps = vid_capture.get(5)
	print('Frames per second : ', fps,'FPS')

	# Get frame count
	# You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
	frame_count = vid_capture.get(7)
	print('Frame count : ', frame_count)

while(vid_capture.isOpened()):
	# vid_capture.read() methods returns a tuple, first element is a bool 
	# and the second is frame
	ret, frame = vid_capture.read()
	if ret == True:
		cv2.imshow('Frame',frame)
		# 20 is in milliseconds, try to increase the value, say 50 and observe
		key = cv2.waitKey(20)
		
		if key == ord('q'):
			break
	else:
		break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()