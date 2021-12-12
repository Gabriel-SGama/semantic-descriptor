#!/bin/bash
# source "/home/gama/venv/bin/activate"
source /media/lasi/disks/documentos/gama/venv/bin/activate

python3 ../kerasMain.py ../args/kitti_ft1.txt

python3 ../kerasMain.py ../args/kitti_ft2.txt

python3 ../kerasMain.py ../args/kitti_ft3.txt

python3 ../kerasMain.py ../args/kitti_ft4.txt

python3 ../kerasMain.py ../args/kitti_ft5.txt

python3 ../kerasMain.py ../args/kitti_ft6.txt

