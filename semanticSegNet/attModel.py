import os
import sys
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras.layers import ReLU
# from tensorflow.keras.layers import sigmoid
from tensorflow.keras.activations import sigmoid
# from tensorflow.keras.applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K

def createAttModel(filePath, img_height, img_width):
    # if(filePath != ''):
    #     print('Using load Model')
    #     model = tf.keras.models.load_model(filePath, compile=False)
    #     return model
    
    
    print('Creating new model')

    # s1TruckInput = tf.keras.layers.Input(dtype=tf.float32, shape=(16, 32, 2048), name ='s1TruckInput') #resnet truck input
    # s1Input = tf.keras.layers.Input(dtype=tf.float32, shape=(256,512,35), name ='s1Input') #scale 1
    # s2Input = tf.keras.layers.Input(dtype=tf.float32, shape=(512,1024,35), name ='s2Input') #scale 2

    s1TruckInput = tf.keras.layers.Input(dtype=tf.float32, shape=(int(img_height/32), int(img_width/32), 2048), name ='s1TruckInput') #resnet truck input
    s1Input = tf.keras.layers.Input(dtype=tf.float32, shape=(int(img_height/2),int(img_width/2), 35), name ='s1Input') #scale 1
    s2Input = tf.keras.layers.Input(dtype=tf.float32, shape=(img_height, img_width, 35), name ='s2Input') #scale 2

    x = Conv2DTranspose(516, (3,3), strides=(2,2), padding = 'same', name = 'conv_up_att_1')(s1TruckInput)
    # x = Conv2D(516, (3,3), strides=(1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # print("x1 shape: " + str(x.shape))

    x = Conv2DTranspose(256, (3,3), strides=(2,2), padding = 'same', name = 'conv_up_att_2')(x)
    # x = Conv2D(256, (3,3), strides=(1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # print("x2 shape: " + str(x.shape))

    x = Conv2DTranspose(128, (3,3), strides=(2,2), padding = 'same', name = 'conv_up_att_3')(x)
    # x = Conv2D(128, (3,3), strides=(1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # print("x3 shape: " + str(x.shape))

    x = Conv2DTranspose(64, (3,3), strides=(2,2), padding = 'same', name = 'conv_up_att_4')(x)
    # x = Conv2D(64, (3,3), strides=(1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # print("x4 shape: " + str(x.shape))

    x = Conv2D(1, (1,1), strides=(1,1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = tf.nn.sigmoid(x)
    # print("x5 shape: " + str(x.shape))

    s1SegAtt = s1Input*x
    # s2SegAtt = s2Input*(1-tf.image.resize(x, (512, 1024)))
    # output = tf.image.resize(s1SegAtt, (512, 1024)) + s2SegAtt

    s2SegAtt = s2Input*(1-tf.image.resize(x, (img_height, img_width)))
    output = tf.image.resize(s1SegAtt, (img_height, img_width)) + s2SegAtt

    # s2SegAtt = s2Input*(1-tf.image.resize(x, (x.shape[1], x.shape[2])))
    # output = tf.image.resize(s1SegAtt, (x.shape[1], x.shape[2])) + s2SegAtt

    model = Model([s1TruckInput, s1Input, s2Input], [output,x], name = 'attModel')
    # tf.keras.utils.plot_model(model, "attModel.png", show_shapes=False)

    # model.summary()
    if(filePath != ''):
        print('copying weights')
        modelW = tf.keras.models.load_model(filePath, compile=False)
        model.set_weights(modelW.get_weights()) 
    
    
    return model
