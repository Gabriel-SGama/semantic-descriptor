from glob import glob
import tensorflow as tf
import tensorflow_addons as tfa
import random
import math
import numpy as np

@tf.function
def _parse_function(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, expand_animations = False, channels = 1)

    #label
    label = tf.io.read_file(label)
    label = tf.image.decode_image(label, expand_animations = False, channels = 3)
    
    return image, label

def configure_for_performance(ds, batch_size, AUTOTUNE):
    ds = ds.batch(batch_size)
    ds = ds.prefetch(10)
    return ds

path = '/home/gama/Documents/datasets/mapillary/'

addPathLabel = 'v2.0/instances/*.png'
addPathCLabel = 'v2.0/labels/*.png'

fullPathLabel = path + 'val' + '*/' + addPathLabel
fullPathCLabel = path + 'val' + '*/' + addPathCLabel

print(fullPathLabel)
print(fullPathCLabel)

labelsList = sorted(glob(fullPathLabel))
colorLabelList = sorted(glob(fullPathCLabel))

AUTOTUNE = tf.data.experimental.AUTOTUNE

batch_size = 1

dataset = tf.data.Dataset.from_tensor_slices((labelsList, colorLabelList))
dataset = dataset.map(_parse_function, num_parallel_calls = AUTOTUNE)
dataset = configure_for_performance(dataset, batch_size, AUTOTUNE)


colorPallet = np.zeros((124,3), dtype = int)

for idx, batch in enumerate(dataset):
    labelData, colorLabelData = batch
    for label, Colorlabel in zip(labelData, colorLabelData):
        label_disp = label.numpy()
        label_disp = np.squeeze(label_disp)

        Colorlabel_disp = Colorlabel.numpy()

        colorPallet[label_disp] = Colorlabel_disp
        print(idx)



# for labelLine, colorLine in zip(label_disp, Colorlabel_disp):
    # colorPallet[labelLine] = colorLine

# for labelLine, colorLine in zip(label_disp, Colorlabel_disp):
#     for label, color in zip(labelLine, colorLine):
#         colorPallet[label] = color



print(colorPallet)
# print(label_disp)
# print(Colorlabel_disp)
