import tensorflow as tf
import tensorflow_addons as tfa
import random
import math
import numpy as np
import cv2

from glob import glob

class dataloader:
    
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.pathImages = args.dataset_images_path
        self.pathLabels = args.dataset_labels_path
        self.pathExtraImages = args.dataset_extra_images_path
        self.pathAutoLabels = args.dataset_auto_labels_path
        self.dataset_infer_path = args.dataset_infer_path
        self.dataset_save_infer_path = args.dataset_save_infer_path
        self.dataset = args.dataset
        self.mode = args.mode

    def loadDataset(self):

        print("start loading dataset\n")

    #-----------images path-----------
        # pathImages = args.dataset_images_path

        if(self.dataset == 'cityscapes'):
            addPathImg = '*/*_leftImg8bit.png'
            addPathLabel = '*/*_labelIds.png'
        elif(self.dataset == 'kitti'):
            addPathImg = 'image_2/*.png'
            addPathLabel = 'semantic/*.png'
        else:
            #Mapillary
            addPathImg = 'images/*.jpg'
            addPathLabel = 'v2.0/instances/*.png'

        print(self.pathImages + '/train*/' + addPathImg)
        trainImages = sorted(glob(self.pathImages + '/train*/' + addPathImg))
        valImages = sorted(glob(self.pathImages + '/val*/' + addPathImg))
        testImages = sorted(glob(self.pathImages + '/test/*' + addPathImg))
        
       
    #-----------labels path-----------
        # pathLabels = args.dataset_labels_path

        trainLabels = sorted(glob(self.pathLabels + '/train*/' + addPathLabel))
        valLabels = sorted(glob(self.pathLabels + '/val*/' + addPathLabel))
        testLabels = sorted(glob(self.pathLabels + '/test*/' + addPathLabel))

        if(self.dataset == "cityscapes" and self.pathExtraImages != "" and self.pathAutoLabels != ""):
            trainImages += glob(self.pathExtraImages + '/train_extra/*/*.png')
            trainImages = sorted(trainImages)
            trainLabels += glob(self.pathAutoLabels + '/*/*.png')
            trainLabels = sorted(trainLabels)

        elif(self.dataset == "kitti"):
            valImages = trainImages[int(0.95*len(trainImages)):]
            trainImages = trainImages[0:int(0.95*len(trainImages))]
            valLabels = trainLabels[int(0.95*len(trainLabels)):]
            trainLabels = trainLabels[0:int(0.95*len(trainLabels))]


        # print(trainImages[3500])
        # print(trainLabels[3500])

    #-----------dataset-----------
        sizeDict = {'train' : len(trainImages), 'val' : len(valImages), 'test' : len(testImages)}

        print('tamanho do dataset train: ' + str(len(trainImages)) + ' | ' + str(len(trainLabels)))
        print('tamanho do dataset val: ' + str(len(valImages)) + ' | ' + str(len(valLabels)))
        print('tamanho do dataset test: ' + str(len(testImages)) + ' | ' + str(len(testLabels)))

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        trainDataset = tf.data.Dataset.from_tensor_slices((trainImages, trainLabels))
        trainDataset = trainDataset.map(self._parse_function_data_att, num_parallel_calls = AUTOTUNE) if(self.mode == 'att') else trainDataset.map(self._parse_function_data_aug, num_parallel_calls = AUTOTUNE)
        trainDataset = configure_for_performance(trainDataset, self.batch_size, AUTOTUNE)

        valDataset = tf.data.Dataset.from_tensor_slices((valImages, valLabels))
        valDataset = valDataset.map(self._parse_function, num_parallel_calls = AUTOTUNE)
        valDataset = configure_for_performance(valDataset, self.batch_size, AUTOTUNE)

        # testDataset = tf.data.Dataset.from_tensor_slices((testImages, testLabels))
        # testDataset = testDataset.map(_parse_function, num_parallel_calls = AUTOTUNE)
        # testDataset = configure_for_performance(testDataset, self.batch_size, AUTOTUNE)

        return trainDataset, valDataset, None, sizeDict


    def boundryLabelRelaxation(self, label):
        # Convolve label for boundry relaxation
        kernel = np.ones((3, 3, 1, 1), np.float32) / 9.0
        # kernel = np.expand_dims(kernel, -1)
        # kernel = np.expand_dims(kernel, -1)

        label = tf.expand_dims(label, -1)
        label = tf.nn.depthwise_conv2d(label, kernel, strides=[1,1,1,1], padding='SAME')
        label = tf.squeeze(label, -1)
        
        return label


    @tf.function
    def zoom_image(self, image, label):

        if(tf.shape(image)[0] < 2*self.img_height or tf.shape(image)[1] < 2*self.img_width):
            image = tf.image.resize(image, (2*self.img_height, 2*self.img_width))
            label = tf.image.resize(label, (self.img_height, self.img_width))

        offset_height = tf.random.uniform(shape=[], minval=0, maxval=self.img_height, dtype=tf.int32)
        offset_width = tf.random.uniform(shape=[], minval=0, maxval=self.img_width, dtype=tf.int32)

        image = tf.cast(tf.image.crop_to_bounding_box(image, offset_height, offset_width, self.img_height, self.img_width), tf.float32)

        label = tf.image.crop_to_bounding_box(label, offset_height, offset_width, self.img_height, self.img_width)
        label = tf.image.resize(label, (int(self.img_height/2), int(self.img_width/2)))

        return image, label


    @tf.function
    def read_for_data(self, filename, labelfile):
        #image input
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, expand_animations = False, channels = 3)
        image = tf.cast(image, tf.float32)
        
        #label
        label = tf.io.read_file(labelfile)
        label = tf.image.decode_image(label, expand_animations = False, channels = 1)

        label = tf.cast(tf.one_hot(tf.squeeze(tf.cast(label, tf.uint8), 2), depth=35), tf.float32)
        return image, label

    @tf.function
    def _parse_function_data_aug(self, filename, labelfile):
        image, label = self.read_for_data(filename, labelfile)

        #zoom by crop
        chance_crop = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5
        
        image, label = tf.cond(chance_crop, lambda: (tf.image.resize(image, (self.img_height, self.img_width)), 
                                                     tf.image.resize(label, (int(self.img_height/2), int(self.img_width/2)))), 
                                            lambda: self.zoom_image(image, label))

        #normalization
        image = image/255.

        #color augmentation
        chance_aug_img = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5
        image = tf.cond(chance_aug_img, lambda: image, lambda: augmentImageGBC(image))
        image = 2.*(image-0.5) #-1 to 1


        # Convolve label for boundry relaxation
        # label = self.boundryLabelRelaxation(label)

        #rotation
        chance_rot = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5
        rot_angle =  random.uniform(-0.0872665, 0.0872665)

        image = tf.cond(chance_rot, lambda: image, lambda: rotateImg(image, rot_angle, [self.img_height, self.img_width]))
        label = tf.cond(chance_rot, lambda: label, lambda: rotateImg(label, rot_angle, [int(self.img_height/2), int(self.img_width/2)]))

        #flip
        chance_flip = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5

        image = tf.cond(chance_flip, lambda: image, lambda: tf.image.flip_left_right(image))
        label = tf.cond(chance_flip, lambda: label, lambda: tf.image.flip_left_right(label))

        return image, label


    @tf.function
    def _parse_function_data_att(self, filename, label):
        #image input
        imageS2 = tf.io.read_file(filename)
        imageS2 = tf.image.decode_image(imageS2, expand_animations = False, channels = 3)
        imageS2 = tf.cast(imageS2, tf.float32)/255.

        chance_aug_img = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5
        imageS2 = tf.cond(chance_aug_img, lambda: imageS2, lambda: augmentImageGBC(imageS2))

        imageS2 = 2.*(imageS2-0.5) #-1 to 1
        imageS1 = tf.image.resize(imageS2, (self.img_height, self.img_width)) #scale 1 

        #label
        label = tf.io.read_file(label)
        label = tf.image.decode_image(label, expand_animations = False, channels = 1)
        label = tf.cast(label, tf.uint8)
        label = tf.cast(tf.one_hot(tf.squeeze(tf.cast(label, tf.uint8), 2), depth=35), tf.float32)

        #rotation
        chance_rot = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5
        rot_angle =  random.uniform(-0.0872665, 0.0872665)
        imageS1 = tf.cond(chance_rot, lambda: imageS1, lambda: rotateImg(imageS1, rot_angle, [self.img_height, self.img_width]))
        imageS2 = tf.cond(chance_rot, lambda: imageS2, lambda: rotateImg(imageS2, rot_angle, [self.img_height, self.img_width]))

        label = tf.image.resize(label, (self.img_height, self.img_width))
        label = tf.cond(chance_rot, lambda: label, lambda: rotateImg(label, rot_angle, [self.img_height/2, self.img_width/2]))

        #flip
        chance_flip = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5
        imageS1 = tf.cond(chance_flip, lambda: imageS1, lambda: tf.image.flip_left_right(imageS1))
        imageS2 = tf.cond(chance_flip, lambda: imageS2, lambda: tf.image.flip_left_right(imageS2))
        label = tf.cond(chance_flip, lambda: label, lambda: tf.image.flip_left_right(label))

       
        return imageS1, imageS2, label


    @tf.function
    def loadInferDataset(self):
        print("path to dataset: ", self.dataset_infer_path)
        print("path to save infering: ", self.dataset_save_infer_path)

        img_filename = sorted(glob(self.dataset_infer_path))
        print("size of dataset: ", len(img_filename))

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        inferDataset = tf.data.Dataset.from_tensor_slices((img_filename))
        inferDataset = inferDataset.map(self._parse_function_infer, num_parallel_calls = AUTOTUNE)
        inferDataset = inferDataset.batch(self.batch_size)
        inferDataset = inferDataset.prefetch(self.batch_size)
        
        return inferDataset


    @tf.function
    def _parse_function_infer(self, img_filename):
        imageS1 = tf.io.read_file(img_filename)
        imageS1 = tf.image.decode_image(imageS1, expand_animations = False, channels = 3)

        imageS1 = tf.image.resize(imageS1, (self.img_height, self.img_width))
        imageS1 = imageS1/255.

        # normalization
        imageS1 = 2.*(imageS1-0.5) #-1 to 1

        imageS2 = tf.image.resize(imageS1, (2*self.img_height, 2*self.img_width))

        return imageS1, imageS2, img_filename
    

    @tf.function
    def _parse_function(self, img_filename, label_filename):
        image = tf.io.read_file(img_filename)
        image = tf.image.decode_image(image, expand_animations = False, channels = 3)

        image = tf.image.resize(image, (self.img_height, self.img_width))
        image = image/255.

        # normalization
        image = 2.*(image-0.5) #-1 to 1


        #label
        if(label_filename == ''):
            return image, image
        
        label = tf.io.read_file(label_filename)
        label = tf.image.decode_image(label, expand_animations = False, channels = 1)
        label = tf.image.resize(label, (int(self.img_height/2), int(self.img_width/2)))
        # label = tf.cast(label, tf.uint8)
        label = tf.cast(tf.one_hot(tf.squeeze(tf.cast(label, tf.uint8), 2), depth=35), tf.float32)
        
        return image, label


def configure_for_performance(ds, batch_size, AUTOTUNE):
    ds = ds.shuffle(buffer_size = 16)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(4*batch_size)
    return ds

def augmentImageGBC(image):

#-----------gamma augmentation-----------
    gamma = tf.random.uniform(shape=[], minval=0.9, maxval=1.1, dtype=tf.float32)
    image = image ** gamma

    brightness = tf.random.uniform(shape=[], minval=0.75, maxval=1.25, dtype=tf.float32)
    image = image * brightness

#-----------color augmentation-----------
    colors = tf.random.uniform([3], 0.9, 1.1)
    white = tf.ones([tf.shape(image)[0], tf.shape(image)[1]])
    color_image = tf.stack([white * colors[i] for i in range(3)], axis=2)
    image *= color_image

    #clip
    image = tf.clip_by_value(image,  0, 1)
    
    return image


@tf.function
def _parse_function(img_filename, label_filename, img_height, img_width):
    image = tf.io.read_file(img_filename)
    image = tf.image.decode_image(image, expand_animations = False, channels = 3)

    image = tf.image.resize(image, (img_height, img_width))
    image = image/255.

    # normalization
    image = 2.*(image-0.5) #-1 to 1


    #label
    if(label_filename == ''):
        return image, image
    
    label = tf.io.read_file(label_filename)
    label = tf.image.decode_image(label, expand_animations = False, channels = 1)
    label = tf.image.resize(label, (int(img_height/2), int(img_width/2)))
    label = tf.cast(label, tf.uint8)
    label = tf.cast(tf.one_hot(tf.squeeze(tf.cast(label, tf.uint8), 2), depth=35), tf.float32)
    
    return image, label


def rotateImg(image, rot_angle, size):

    vsin = abs(math.sin(rot_angle))
    vcos = abs(math.cos(rot_angle))

    new_height = tf.cast(size[0]*vcos + size[1]*vsin, tf.int32)
    new_width = tf.cast(size[1]*vcos + size[0]*vsin, tf.int32)

    # print(image.shape)
    image = tf.image.resize(image, (new_height, new_width))
    # print(image.shape)
    # print(type(image))
    # print(rot_angle)
    # image = tf.squeeze(image, 2)
    # print(image.shape)
    # print(type(image))
    # tf.keras.preprocessing.image.random_rotation(
    # image, rot_angle, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest',
    # cval=0.0, interpolation_order=1)
    
    image = tfa.image.rotate(image, rot_angle, interpolation = 'nearest')
    
    offset_height = tf.cast((new_height - size[0])/2, tf.int32)
    offset_width = tf.cast((new_width - size[1])/2, tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, size[0], size[1])
    return image

