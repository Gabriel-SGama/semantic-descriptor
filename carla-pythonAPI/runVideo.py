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
vid_capture = cv2.VideoCapture('out_camera_kitti.avi')
vid_seg = cv2.VideoCapture('out_seg_kitti.avi')

if (vid_capture.isOpened() == False or vid_seg.isOpened() == False):
	print("Error opening the video file")

else:
	# Get frame rate information
	fps = vid_capture.get(cv2.CAP_PROP_FPS)
	print('Frames per second : ', fps,'FPS')

	# Get frame count
	frame_count = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
	print('Frame count : ', frame_count)

i=0
while(vid_capture.isOpened() and vid_seg.isOpened()):
	# vid_capture.read() methods returns a tuple, first element is a bool 
	# and the second is frame
	ret, frame = vid_capture.read()
	ret_seg, frame_seg = vid_seg.read()
	if ret == True:

		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		cv2.imshow('camera',frame)
		cv2.imshow('seg',frame_seg)
		cv2.imwrite('image_2/' +str(i).zfill(6) + ".png", frame)

		# 20 is in milliseconds, try to increase the value, say 50 and observe
		key = cv2.waitKey(20)
		
		if key == ord('q'):
			break
		i+=1
	else:
		break

# Release the video capture object
vid_capture.release()
vid_seg.release()
cv2.destroyAllWindows()