import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
# tf.keras.applications.ResNet50V2

import matplotlib.pyplot as plt
import numpy as np

import tensorflow.keras as keras
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
# from tensorflow.keras.applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K


def resNet50V2():
    model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000)
    return model

def batchRelu(x, name):
    x = BatchNormalization()(x)
    return ReLU()(x)

def resNet50V2Dec(input):
    x = Conv2DTranspose(1024, (3,3), strides=(2,2), padding = 'same', name = 'conv_up_1')(input)
    x = batchRelu(x, 'bn_up_1')
    x = Conv2D(512, (3,3), strides=(1,1), padding = 'same')(x)
    x = batchRelu(x, 'bn_conv_1')
    
    # x = BatchNormalization(name = 'bn_up_1')(x)
    # x = ReLU()(x)

    x = Conv2DTranspose(512, (3,3), strides=(2,2), padding = 'same', name = 'conv_up_2')(x)
    x = batchRelu(x, 'bn_up_2')
    x = Conv2D(256, (3,3), strides=(1,1), padding = 'same')(x)
    x = batchRelu(x, 'bn_conv_2')
    
    # x = BatchNormalization(name = 'bn_up_2')(x)
    # x = ReLU()(x)

    x = Conv2DTranspose(256, (3,3), strides=(2,2), padding = 'same', name = 'conv_up_3')(x)
    x = batchRelu(x, 'bn_up_3')
    x = Conv2D(128, (3,3), strides=(1,1), padding = 'same')(x)
    x = batchRelu(x, 'bn_conv_3')
    
    # x = BatchNormalization(name = 'bn_up_3')(x)
    # x = ReLU()(x)

    x = Conv2DTranspose(128, (3,3), strides=(2,2), padding = 'same', name = 'conv_up_4')(x)
    x = batchRelu(x, 'bn_up_4')
    x = Conv2D(64, (3,3), strides=(1,1), padding = 'same')(x)
    x = batchRelu(x, 'bn_conv_4')
    
    # x = BatchNormalization(name = 'bn_up_4')(x)
    # x = ReLU()(x)

    x = Conv2D(35, (1,1), strides=(1,1), padding = 'same', name='conv_final')(x)
    x = tf.nn.softmax(x)
    return x

def resNet50V2Model(filePath):
    if(filePath != ''):
        print('Using pre trained model - Mapillary')
        base_model = keras.models.load_model(filePath, compile=False)
    else:
        print('Using pre trained model - Imagenet')
        base_model = resNet50V2()

    model = Model(base_model.input, base_model.get_layer('post_relu').output, name = 'resnet50v2_variation')
    output = resNet50V2Dec(model.output)
    model = Model(model.input, output, name = 'resnet50v2_variation_dec')
    # model.summary()
    
    return model

