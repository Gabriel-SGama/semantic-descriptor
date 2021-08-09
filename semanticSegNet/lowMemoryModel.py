import warnings
import numpy as np

import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
import os
import time
from glob import glob

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
from tensorflow.keras.activations import tanh

# from tensorflow.keras.applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K


def inceptionBlock(input, channels, blockNumber, strides = (1,1)):

    x_1 = Conv2D(channels, (1,1), strides=strides, padding='same', name = 'incp_block_' + str(blockNumber) + '_conv_1_1')(input)
    x_1 = Conv2D(channels, (3,3), strides=(1,1), padding='same', name = 'incp_block_' + str(blockNumber) + '_conv_1_2')(x_1)


    x_2 = Conv2D(channels, (1,1), strides=strides, padding='same', name = 'incp_block_' + str(blockNumber) + '_conv_2_1')(input)
    x_2 = Conv2D(channels, (5,5), strides=(1,1), padding='same', name = 'incp_block_' + str(blockNumber) + '_conv_2_2')(x_2)


    x_3 = MaxPooling2D((3,3), strides=(1,1), padding='same', name = 'incp_block_' + str(blockNumber) + '_max_pool')(input)
    x_3 = Conv2D(2*channels, (1,1), strides=strides, padding='same', name = 'incp_block_' + str(blockNumber) + '_conv_3_1')(x_3)


    x = tf.keras.layers.concatenate([x_1, x_2, x_3], axis = 3)
    return x

def LMM():
    input_img = Input(shape = (256, 512, 3))

    x = Conv2D(64, (7,7), strides=(1,1), padding = 'same', name = 'conv_1')(input_img)
    x = tanh(x)

    x = Conv2D(128, (5,5), strides=(1,1), padding = 'same', name = 'conv_2')(x)
    x = tanh(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None, name = 'max_pool_1')(x)
    
    x = Conv2D(128, (3,3), strides=(1,1), padding = 'same', name = 'conv_3')(x)
    x = tanh(x)
    
    x = inceptionBlock(x, 32, 1, strides = (2,2)) #shape/2
    # x = inceptionBlock(x, 128, 2)

    x = Conv2D(128, (3,3), strides=(1,1), padding = 'same', name = 'conv_4')(x)
    x = tanh(x)
    
    x = Conv2DTranspose(128, (3,3), strides=(2,2), padding = 'same', name = 'conv_up_1')(x) #shape/2
    x = tanh(x)

    x = Conv2DTranspose(128, (3,3), strides=(2,2), padding = 'same', name = 'conv_up_2')(x) #shape/2
    x = tanh(x)

    x = Conv2D(64, (3,3), strides=(1,1), padding = 'same', name = 'conv_5')(x)
    x = tanh(x)

    x = Conv2D(35, (1,1), strides=(1,1), padding = 'same', name = 'conv_6')(x)
    x = tf.nn.softmax(x)
    
    model = Model(inputs = input_img,outputs = x, name = 'LMM')
    
    # # model = tf.keras.Model(inputs = input_s2, outputs = segOut)
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss = tf.keras.losses.categorical_crossentropy(),
    #             metrics=['accuracy'])

    model.summary()
    return model