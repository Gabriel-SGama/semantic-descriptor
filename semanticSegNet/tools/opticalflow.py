import math as m
import cv2 as cv
import numpy as np
from glob import glob
from tqdm import tqdm
import sys

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

filter1 = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

filter2 = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])

# filter2D() function can be used to apply kernel to an image.
# Where ddepth is the desired depth of final image. ddepth is -1 if...
# ... depth is same as original or source image.


# # Parameters for ShiTomasi corner detection
# feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=3, blockSize=10)

# # Parameters for Lucas Kanade optical flow
# lk_params = dict(
#     winSize=(15, 15),
#     maxLevel=2,
#     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
# )

optical_flow = cv.optflow.DualTVL1OpticalFlow_create(nscales=8,epsilon=0.05,warps=4)

seq = "/home/gama/Documentos/datasets/kitti/data_odometry_color/dataset/sequences/03/image_2/*.png"
sem_seq = "/home/gama/Documentos/datasets/kitti/data_odometry_color/dataset/sequences/03/semantic/label/*.png"
color_seq = "/home/gama/Documentos/datasets/kitti/data_odometry_color/dataset/sequences/03/semantic/color/*.png"
alFiles = sorted(glob(seq))
alSemFiles = sorted(glob(sem_seq))
alColorFiles = sorted(glob(color_seq))
print("rgb", len(alFiles))
print("label", len(alSemFiles))
print("color", len(alColorFiles))

prev = cv.imread(alFiles[0], cv.IMREAD_UNCHANGED)
sem_prev = cv.imread(alSemFiles[0], cv.IMREAD_UNCHANGED)
color_prev = cv.imread(alColorFiles[0], cv.IMREAD_UNCHANGED)
color_prev = cv.resize(color_prev, (prev.shape[1], prev.shape[0]), interpolation = cv.INTER_AREA)

alFiles = alFiles[1:11]
alSemFiles = alSemFiles[1:11]
alColorFiles = alColorFiles[1:11]

# Creates an image filled with zero
# intensities with the same dimensions 
# as the frame
mask = np.zeros_like(prev)
  
# Sets image saturation to maximum
mask[..., 1] = 255
prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
orb = cv.ORB_create()

for file, semFile, colorFile in tqdm(zip(alFiles, alSemFiles, alColorFiles)):
        
    frame = cv.imread(file, cv.IMREAD_UNCHANGED)
    sem = cv.imread(semFile, cv.IMREAD_UNCHANGED)
    sem = cv.resize(sem, (frame.shape[1], frame.shape[0]), interpolation = cv.INTER_AREA)
    color = cv.imread(colorFile, cv.IMREAD_UNCHANGED)
    color = cv.resize(color, (frame.shape[1], frame.shape[0]), interpolation = cv.INTER_AREA)

    # Opens a new window and displays the input
    # frame
    cv.imshow("input", frame)

    # Converts each frame to grayscale - we previously 
    # only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    print("frame: ", frame.shape)
    print("sem: ", sem.shape)

    # semf1 = cv.filter2D(src=sem, ddepth=-1, kernel=filter1)
    # semf2 = cv.filter2D(src=sem, ddepth=-1, kernel=filter2)
    # semf = cv.add(semf1, semf2)
    # image = np.zeros_like(semf)
    # image[semf > 0] = 255
    # cv.imshow("semf", image)
    
    # colorFlow = np.random.randint(0, 255, (100, 3))

    # # queryKeypoints, queryDescriptors = orb.detectAndCompute(sem_prev,None)
    # # p0 = queryKeypoints
    # p0 = cv.goodFeaturesToTrack(sem_prev, mask=None, **feature_params)
    # print(p0)
    # # Create a mask image for drawing purposes
    # mask = np.zeros_like(prev)

    # # Calculate Optical Flow
    # p1, st, err = cv.calcOpticalFlowPyrLK(
    #     prev, gray, p0, None, **lk_params
    # )
    # # Select good points
    # good_new = p1[st == 1]
    # good_old = p0[st == 1]

    # # Draw the tracks
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     a = int(a)
    #     b = int(b)
    #     c = int(c)
    #     d = int(d)

    #     mask = cv.line(mask, (a, b), (c, d), colorFlow[i].tolist(), 2)
    #     gray = cv.circle(gray, (a, b), 5, colorFlow[i].tolist(), -1)

    # # Display the demo
    # img = cv.add(gray, mask)
    # cv.imshow("frame", img)

    # # Update the previous frame and previous points
    # prev = gray.copy()
    # p0 = good_new.reshape(-1, 1, 2)

    # Calculates dense optical flow by Farneback method
    # flow = cv.calcOpticalFlowFarneback(prev, gray, 
    #                                    None,
    #                                    0.5, 3, 15, 3, 5, 1.2, 0)

    flow = optical_flow.calc(prev, gray, None)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    pred_color = color_prev.copy()
    # pred_color = np.zeros_like(color_prev)
    print(pred_color.shape)
    print(magnitude.shape)
    print(angle.shape)

    # offset_x = np.zeros_like(sem)
    # offset_y = np.zeros_like(sem)
    offset_x = np.rint(np.multiply(magnitude,np.cos(angle))).astype(int)
    offset_y = np.rint(np.multiply(magnitude,np.sin(angle))).astype(int)

    # Rows = np.arange(0, magnitude.shape[0], 1)
    # nRows = np.repeat(Rows, magnitude.shape[1]).reshape(magnitude.shape)


    Rows = np.arange(0, magnitude.shape[0], 1)
    nRows = np.zeros_like(sem, dtype=np.int32)
    for i in range(nRows.shape[1]):
        nRows[:,i] = Rows
   
    # np.set_printoptions(threshold=sys.maxsize)

    Cols = np.arange(0, magnitude.shape[1], 1)
    nCols = np.zeros_like(sem, dtype=np.int32)
    for i in range(nCols.shape[0]):
        nCols[i,:] = Cols
    
    ofx = nCols + offset_x
    ofy = nRows + offset_y
    
    result_x = np.zeros_like(sem)
    result_y = np.zeros_like(sem)

    print("nCols: ", nCols.shape)
    print("nRows: ", nRows.shape)
    print("result_x: ", result_x.shape)
    print("result_y: ", result_y.shape)
    print("ofx: ", ofx.shape)
    print("ofy: ", ofy.shape)

    result_x = np.where(np.logical_and(ofx > 0, ofx <  magnitude.shape[1]-1), ofx, nCols)
    result_y = np.where(np.logical_and(ofy > 0, ofy <  magnitude.shape[0]-1), ofy, nRows)



    print("result_x: ", result_x.shape)
    print("result_y: ", result_y.shape)
    pred_color[result_y, result_x] = color_prev

    # for row in range(magnitude.shape[0]):
    #     for col in range(magnitude.shape[1]):
    #         # print("row: ", row)
    #         # print("col: ", col)
    #         mag = magnitude[row, col]
    #         ang = angle[row, col]
    #         new_row = int(np.rint(row + mag*m.sin(ang)))
    #         new_col = int(np.rint(col + mag*m.cos(ang)))
    #         # print("new row: ", new_row)
    #         # print("new col: ", new_col)
    #         if(new_row > 0 and new_row < magnitude.shape[0]-1 and new_col > 0 and new_col < magnitude.shape[1]-1):
    #             pred_color2[new_row, new_col] = color_prev[row, col]

    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow
    # magnitude (normalized)
    # print(magnitude)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)
    cv.imshow("semantic label", color)
    cv.imshow("prev semantic label", color_prev)
    cv.imshow("pred semantic label", pred_color)

    # Updates previous frame
    prev = gray
    sem_prev = sem
    color_prev = color
    # Frames are read by intervals of 1 millisecond. The
    # programs breaks out of the while loop when the
    # user presses the 'q' key
    if cv.waitKey(0) & 0xFF == ord('q'):
        break
  
# The following frees up resources and
# closes all windows
cv.destroyAllWindows()
