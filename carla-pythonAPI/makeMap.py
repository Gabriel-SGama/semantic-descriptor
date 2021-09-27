from operator import pos
import cv2 
import numpy as np
from numpy.core import machar
import re

# with open('/home/gama/code/semantic-descriptor/slam/data.txt') as f:
with open('pos.txt') as f:
    lines = f.readlines()

_WIDTH = 512
_HEIGHT = 512
position = np.zeros((len(lines), 6), np.float)
map = np.zeros((_WIDTH,_HEIGHT,3), np.uint8)
cv2.namedWindow('Map', cv2.WINDOW_AUTOSIZE)

# initcial_pos = lines[0]
# print(initcial_pos)
# for line in lines:

# match = re.match(r"x:  (\d*\.\d+)|d+", str(initcial_pos))
# match = re.match(r"x:([-]?\d*\.\d+), y:([-]?\d*\.\d+), z:([-]?\d*\.\d+), roll:([-]?\d*\.\d+), pitch:([-]?\d*\.\d+), yaw:([-]?\d*\.\d+)", initcial_pos)

# x = float(match.group(1))
# y = float(match.group(2))
# z = float(match.group(3))
# roll = float(match.group(4))
# pitch = float(match.group(5))
# yaw = float(match.group(6))

# position[0] = [x,y,z,roll, pitch, yaw]
idx = 0

max_x = 100
max_y = 100
max_z = 100
roll = 0
pitch = 0
yaw = 0

x_ant = 0
y_ant = 0
z_ant = 0

for line in lines:
    match = re.match(r"x:([-]?\d*\.\d+), y:([-]?\d*\.\d+), z:([-]?\d*\.\d+), roll:([-]?\d*\.\d+), pitch:([-]?\d*\.\d+), yaw:([-]?\d*\.\d+)", line)
    x = float(match.group(1))
    y = float(match.group(2))
    z = float(match.group(3))
    
    # roll = float(match.group(4))
    # pitch = float(match.group(5))
    # yaw = float(match.group(6))

    position[idx] = [x ,y ,z ,roll, pitch, yaw]

    max_x = max(max_x, abs(x - position[0][0]))
    max_y = max(max_y, abs(y - position[0][1]))
    max_z = max(max_z, abs(z))

    map = np.zeros((_WIDTH,_HEIGHT,3), np.uint8)
    for i in range(idx):
        x = position[i][0]
        y = position[i][1]
        cv2.circle(map, ((int((x-position[0][0])*(_WIDTH/(2.5*max_x)) + _WIDTH/2)), int((y-position[0][1])*(_HEIGHT/(2.5*max_y))+ _HEIGHT/2)), 1, (255,255,255), cv2.FILLED)

    cv2.imshow("Map", map)
    cv2.waitKey(5)
    idx += 1

cv2.waitKey(0)
