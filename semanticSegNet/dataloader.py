import tensorflow as tf
import tensorflow_addons as tfa
import random
import math
from glob import glob

class dataloader:
    
    
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.pathImages = args.dataset_images_path
        self.pathLabels = args.dataset_labels_path
        self.dataset = args.dataset
        self.mode = args.mode

    def loadDataset(self):

        # batch_size = args.batch_size
        # img_width =  args.img_width
        # img_height =  args.img_height

        print("start loading dataset\n")

    #-----------images path-----------
        # pathImages = args.dataset_images_path

        if(self.dataset == 'cityscapes'):
            addPathImg = '*/*_leftImg8bit.png'
            addPathLabel = '*/*_labelIds.png'
        else:
            #Mapillary
            addPathImg = 'images/*.jpg'
            addPathLabel = 'v2.0/instances/*.png'

        print(self.pathImages + 'train*/' + addPathImg)
        trainImages = sorted(glob(self.pathImages + 'train*/' + addPathImg))
        valImages = sorted(glob(self.pathImages + 'val*/' + addPathImg))
        testImages = sorted(glob(self.pathImages + 'test*/' + addPathImg))

    #-----------labels path-----------
        # pathLabels = args.dataset_labels_path

        trainLabels = sorted(glob(self.pathLabels + 'train*/' + addPathLabel))
        valLabels = sorted(glob(self.pathLabels + 'val*/' + addPathLabel))
        testLabels = sorted(glob(self.pathLabels + 'test*/' + addPathLabel))
        
        # print(trainImages[2000])
        # print(trainLabels[2000])

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
        valDataset = valDataset.map(self._parse_function_data_att, num_parallel_calls = AUTOTUNE) if(self.mode == 'att') else valDataset.map(self._parse_function_data_aug, num_parallel_calls = AUTOTUNE)
        valDataset = configure_for_performance(valDataset, self.batch_size, AUTOTUNE)

        # testDataset = tf.data.Dataset.from_tensor_slices((testImages, testLabels))
        # testDataset = testDataset.map(_parse_function, num_parallel_calls = AUTOTUNE)
        # testDataset = configure_for_performance(testDataset, self.batch_size, AUTOTUNE)

        return trainDataset, valDataset, None, sizeDict

    @tf.function
    def _parse_function_data_aug(self, filename, label):
        #image input
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, expand_animations = False, channels = 3)

        # image = tf.image.resize(image, (1024, 2048)) #for mapillary

        #Zoom (by crop)    
        chance_crop = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5

        offset_height = tf.random.uniform(shape=[], minval=0, maxval=self.img_height, dtype=tf.int32)
        offset_width = tf.random.uniform(shape=[], minval=0, maxval=self.img_width, dtype=tf.int32)

        image = tf.cond(chance_crop, lambda: tf.image.resize(image, (self.img_height, self.img_width)), 
            lambda: tf.cast(tf.image.crop_to_bounding_box(image, offset_height, offset_width, self.img_height, self.img_width), tf.float32))

        # image = tf.cast(image, tf.float32)/255.
        image = image/255.

        chance_aug_img = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5
        image = tf.cond(chance_aug_img, lambda: image, lambda: augmentImageGBC(image))

        image = 2.*(image-0.5) #-1 to 1

        #label
        label = tf.io.read_file(label)
        label = tf.image.decode_image(label, expand_animations = False, channels = 1)
        # label = tf.image.resize(label, (1024, 2048)) #for mapillary

        label = tf.cast(label, tf.uint8)
        label = tf.cast(tf.one_hot(tf.squeeze(tf.cast(label, tf.uint8), 2), depth=35), tf.float32)

        #Zoom (by crop)
        label = tf.cond(chance_crop, lambda: label, 
            lambda: tf.image.crop_to_bounding_box(label, offset_height, offset_width,  self.img_height, self.img_width))

        #rotation
        chance_rot = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5
        rot_angle =  random.uniform(-0.0872665, 0.0872665)
        image = tf.cond(chance_rot, lambda: image, lambda: rotateImg(image, rot_angle, [self.img_height, self.img_width]))

        label = tf.image.resize(label, (int(self.img_height/2), int(self.img_width/2)))

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
        label = tf.cond(chance_rot, lambda: label, lambda: rotateImg(label, rot_angle, [self.img_height, self.img_width]))

        #flip
        chance_flip = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) > 0.5
        imageS1 = tf.cond(chance_flip, lambda: imageS1, lambda: tf.image.flip_left_right(imageS1))
        imageS2 = tf.cond(chance_flip, lambda: imageS2, lambda: tf.image.flip_left_right(imageS2))
        label = tf.cond(chance_flip, lambda: label, lambda: tf.image.flip_left_right(label))

        return imageS1, imageS2, label



def configure_for_performance(ds, batch_size, AUTOTUNE):
    ds = ds.shuffle(buffer_size = 16)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
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
    
    # image = tfa.image.rotate(image, rot_angle, interpolation = 'nearest')
    
    offset_height = tf.cast((new_height - size[0])/2, tf.int32)
    offset_width = tf.cast((new_width - size[1])/2, tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, size[0], size[1])
    return image

