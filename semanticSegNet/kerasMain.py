import sys
import argparse
from PIL.Image import new
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
import os
import time
from glob import glob
import random

from tensorflow.python.keras.engine import training
from PIL import Image

#local imports
import dataloader as dl
import labels as lb
import tensorflow.keras as keras
from resnet import *
from resnet50v2 import *
from lowMemoryModel import *
from attModel import *
from tensorflow.keras import backend as K

from twilio.rest import Client


# ============== #
#  Args Parsing  #
# ============== #
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='SemSeg TensorFlow 2 implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--dataset', type=str, help='mapillary or cityscapes', required=True)
parser.add_argument('--dataset_images_path', type=str, help='image path', required=True)
parser.add_argument('--dataset_labels_path', type=str, help='label path', default="")
parser.add_argument('--dataset_infer_path', type=str, help='infer path', default="")
parser.add_argument('--dataset_save_infer_path', type=str, help='save infer path', default="")
parser.add_argument('--img_width', type=int, help='image width', required=True)
parser.add_argument('--img_height', type=int, help='image height', required=True)
parser.add_argument('--num_epochs', type=int, help='number of epochs of training', required=True)
parser.add_argument('--batch_size', type=int, help='batch size', required=True)
parser.add_argument('--GPU', type=str, help='GPU number', required=True)
parser.add_argument('--save_model_path', type=str, help='directory where to save model', default="")
parser.add_argument('--pre_train_model_path', type=str, help='directory to load pre trained model on Mapillary', default="")
parser.add_argument('--load_model_path', type=str, help='directory where to load model from', default="")
parser.add_argument('--load_att_path', type=str, help='directory where to load attention model from', default="")
parser.add_argument('--metrics_path', type=str, help='directory where to save metrics from train and loss', default="test")

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


# =================== #
#  Tensorflow Config  #
# =================== #
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU  # chooses GPU

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate, 75000, end_learning_rate=0.0007, power=2.0,
    cycle=False, name=None
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')


def loss(pred, target):
    return tf.losses.categorical_crossentropy(target, pred)


def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


def makePredImg(pred):
    pred = np.squeeze(pred)

    pred = lb.cityscapes_pallete_float[np.argmax(pred, axis=-1), :]  
    pred = pred[:,:,0:3]

    return pred

def readFiles(dataset, index):
    pathImages = args.dataset_images_path
    pathLabels = args.dataset_labels_path
    width = args.img_width
    height = args.img_height
    
    if(args.dataset == 'cityscapes'):
        addPathImg = '*/*_leftImg8bit.png'
        addPathLabel = '*/*_labelIds.png'
    elif(args.dataset == 'kitti'):
        addPathImg = 'image_2/*.png'
        addPathLabel = 'semantic/*.png'
    else:
        addPathImg = '/images/*.jpg'
        addPathLabel = '/v2.0/instances/*.png'

    print(pathImages + dataset + '*/' + addPathImg)
    if(pathLabels == ''):
        image, label = dl._parse_function(sorted(glob(pathImages + dataset + '*/' + addPathImg))[index], '', height, width)
    else:
        image, label = dl._parse_function(sorted(glob(pathImages + dataset + '*/' + addPathImg))[index], 
                                        sorted(glob(pathLabels + dataset + '*/' + addPathLabel))[index],
                                        height, width)

    return image, label

def predictRand(model, dataset, index, filename='eval_truck'):

    image, label = readFiles(dataset, index)
    image_disp = image/2. + 0.5 #0 to 1 (plot)

    if(args.dataset_labels_path != ''):
        label_disp = label.numpy()
        print("label shape: " + str(label_disp.shape))
        label_disp = lb.cityscapes_pallete_float[np.argmax(label_disp, axis=-1), :] #label for diplay  
        label_disp = label_disp[:,:,0:3]
    else:
        label_disp = image_disp
    
    image_input = np.expand_dims(image, axis=0)

    pred, _ = model.predict(image_input)
    print("pred shape: " + str(pred.shape))

    pred_disp = makePredImg(pred)

    iou = iou_coef(np.expand_dims(label, axis=0), pred)
    print("iou: " + str(iou))
    imgList = []
    imgList.append({'title' : 'Original', 'img' : image_disp})
    imgList.append({'title' : 'Label', 'img' : label_disp})
    imgList.append({'title' : 'Final Pred', 'img' : pred_disp})
    
    displayImage(imgList, image_disp, label_disp, pred_disp, filename)
        
def predictAttention(model, attModel, dataset, index = 10, filename = 'testAttFull'):

    image, label = readFiles(index)

    label = tf.image.resize(label, (512, 1024)) #testing
    
    label_disp = label.numpy()
    label_disp = lb.cityscapes_pallete_float[np.argmax(label_disp, axis=-1), :] #label for diplay  
    label_disp = label_disp[:,:,0:3]
    
    image_disp = image/2. + 0.5 #0 to 1 (plot)

    image_inputS1 = np.expand_dims(image, axis=0) 
    image_inputS2 = tf.image.resize(image_inputS1, (1024, 2048)) #testing

    
    #run model
    predS1, predTruckS1 = model([image_inputS1], training=False)
    predS2, predTruckS2 = model([image_inputS2], training=False)
    
    finalPred = attModel([predTruckS1, predS1, predS2], training = False)

    # attMask = np.squeeze(attMask)

    finalPred_disp = np.squeeze(finalPred)
    finalPred_disp = lb.cityscapes_pallete_float[np.argmax(finalPred_disp, axis=-1), :]  
    finalPred_disp = finalPred_disp[:,:,0:3]
    
    #metric
    iou = iou_coef(np.expand_dims(tf.image.resize(label, (256, 512)), axis=0), predS1)
    iouAtt = iou_coef(np.expand_dims(label, axis=0), finalPred)
    print("\niouS1: " + str(iou))
    print("\niouAtt: " + str(iouAtt))

    #display image
    predS1_disp = makePredImg(predS1)
    predS2_disp = makePredImg(predS2)

    imgList = []
    imgList.append({'title' : 'Original', 'img' : image_disp})
    imgList.append({'title' : 'Label', 'img' : label_disp})
    imgList.append({'title' : 'PredS1', 'img' : predS1_disp})
    imgList.append({'title' : 'PredS2', 'img' : predS2_disp})
    # imgList.append({'title' : 'Att Mask', 'img' : attMask})
    imgList.append({'title' : 'Final Pred Att', 'img' : finalPred_disp})

    displayImage(imgList, image_disp, label_disp, finalPred_disp, filename)

def displayImage(imgList, image, label, finalPred, filename = "test.png"):
    fig = plt.figure(figsize=(15, 15))

    nColuns = 3
    nLines =  int(np.ceil(len(imgList)/3.0))

    for index, img in enumerate(imgList):
        fig.add_subplot(nLines, nColuns, index + 1)
        plt.imshow(img['img'], interpolation='bilinear')
        plt.title(img['title'])
        
    plt.savefig(filename)


def trainTruck(model, trainDataset, valDataset, sizes):

    logdir = "logs/scalars/" + args.metrics_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer_train = tf.summary.create_file_writer(logdir + "_train")
    file_writer_val = tf.summary.create_file_writer(logdir + "_val")

    num_epochs = args.num_epochs

    num_batches = sizes['train'] // args.batch_size
    
    i = 0
    iou_sum = 0
    iou_sum_max = 0
    train_sum_loss_min = 1000
    for e in range(num_epochs):
        train_sum_loss = 0
        batchIndex = 0

        for batchData in trainDataset:
            
            print('\rEpoch {}/{} | batch {}/{} {}{} loss: {}'.format(e + 1, num_epochs, batchIndex, num_batches, '>'*int(batchIndex/50), '-'*int((num_batches-batchIndex)/50), train_sum_loss), end='', flush=True)
            
            images, labels = batchData
            
            # train_loss = train(model, images, labels)
            with tf.GradientTape() as tape:
                pred, inter_out = model(images, training = True)        
                train_loss = loss(pred, labels)

            grads = tape.gradient(train_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_sum_loss = tf.reduce_mean(train_loss)

            with file_writer_train.as_default():
                tf.summary.scalar(f'loss', data=train_sum_loss, step=i)
                tf.summary.scalar(f'lr', data=optimizer._decayed_lr(tf.float32), step=i)
            
            i += 1
            batchIndex += 1

        if(args.dataset != 'kitti'):
            #evaluate validations set
            val_sum_loss = 0
            iou_sum = 0
            for batchData in valDataset:
                images, labels = batchData
                pred, _ = model(images, training = False)
                iou = iou_coef(labels, pred)
    
                val_loss = loss(pred, labels)
                iou_sum += iou
                
                val_sum_loss += tf.reduce_mean(val_loss)

            val_sum_loss /= (sizes['val']/args.batch_size)
            iou_sum /= (sizes['val']/args.batch_size)
            
            with file_writer_val.as_default():
                tf.summary.scalar(f'loss', data=val_sum_loss, step=i)
                tf.summary.scalar(f'iou', data=iou_sum, step=i)

            #saves model if is the best result and it is 60% complete
            if(iou_sum > iou_sum_max):
                iou_sum_max = iou_sum

                if(e > 0.6*num_epochs and args.save_model_path != ""):
                    print("Saving model in " + args.save_model_path + "...")
                    model.save(args.save_model_path)
        
        else:
            if(train_sum_loss < train_sum_loss_min):
                train_sum_loss_min = train_sum_loss

                if(e > 0.6*num_epochs and args.save_model_path != ""):
                    print("Saving model in " + args.save_model_path + "...")
                    model.save(args.save_model_path)
    
    print("EPOCHS = " + str(num_epochs))
    print("IOU MAX = " + str(iou_sum_max))


def trainAtt(model, attModel, trainDataset, valDataset, sizes): #simplify later

    logdir = "logs/scalars/" + args.metrics_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer_train = tf.summary.create_file_writer(logdir + "_train")
    file_writer_val = tf.summary.create_file_writer(logdir + "_val")

    num_epochs = args.num_epochs

    num_batches = sizes['train'] // args.batch_size
    
    i = 0
    iou_sum = 0
    iou_sum_max = 0
    for e in range(num_epochs):
        train_sum_loss = 0
        batchIndex = 0

        for batchData in trainDataset:
            
            print('\rEpoch {}/{} | batch {}/{} {}{} loss: {}'.format(e + 1, num_epochs, batchIndex, num_batches, '>'*int(batchIndex/50), '-'*int((num_batches-batchIndex)/50), train_sum_loss), end='', flush=True)
            
            imageS1, imageS2, label = batchData
            
            #images before training
            predS1, predTruckS1 = model([imageS1], training=False)
            predS2, predTruckS2 = model([imageS2], training=False)
    
            
            # predTruckS1 = truck([imageS1])[0]
            # predS1 = segHead([predTruckS1])[0]
            # predTruckS2 = truck([imageS2])[0]
            # predS2 = segHead([predTruckS2])[0]
            
            with tf.GradientTape() as tape:
                finalPred = attModel([predTruckS1, predS1, predS2], training = True)
                train_loss = loss(finalPred, label)

            grads = tape.gradient(train_loss, attModel.trainable_weights)
            optimizer.apply_gradients(zip(grads, attModel.trainable_weights))

            train_sum_loss = tf.reduce_mean(train_loss)

            with file_writer_train.as_default():
                tf.summary.scalar(f'loss', data=train_sum_loss, step=i)
                tf.summary.scalar(f'lr', data=optimizer._decayed_lr(tf.float32), step=i)
            
            i += 1
            batchIndex += 1


        #evaluate validations set
        val_sum_loss = 0
        iou_sum = 0
        for batchData in valDataset:
            imageS1, imageS2, label = batchData
            
            #images before training
            predS1, predTruckS1 = model([imageS1], training=False)
            predS2, predTruckS2 = model([imageS2], training=False)
    
            # predTruckS1 = truck([imageS1])[0]
            # predS1 = segHead([predTruckS1])[0]
            # predTruckS2 = truck([imageS2])[0]
            # predS2 = segHead([predTruckS2])[0]
            
            # finalPred, attMask = attModel([predTruckS1, predS1, predS2], training = False)
            finalPred = attModel([predTruckS1, predS1, predS2], training = False)
            iou = iou_coef(label, finalPred)
 
            val_loss = loss(finalPred, label)
            iou_sum += iou
            
            val_sum_loss += tf.reduce_mean(val_loss)

        val_sum_loss /= (sizes['val']/args.batch_size)
        iou_sum /= (sizes['val']/args.batch_size)
        
        with file_writer_val.as_default():
            tf.summary.scalar(f'loss', data=val_sum_loss, step=i)
            tf.summary.scalar(f'iou', data=iou_sum, step=i)

        #saves model if is the best result and it is 60% complete
        if(e > 0.6*num_epochs and args.save_model_path != "" and iou_sum > iou_sum_max):
            print("Saving model in " + args.save_model_path + "...")
            attModel.save(args.save_model_path)
            iou_sum_max = iou_sum

    # sendMessage(iou_sum_max, num_epochs)

def inferData(model, attModel, inferDataset):
    

    firstLogOdd = np.zeros((args.img_height, args.img_width, 35), dtype=np.float) #?
    previusLogOdd = np.zeros((args.img_height, args.img_width, 35), dtype=np.float)
    currentLogOdd = np.zeros((args.img_height, args.img_width, 35), dtype=np.float)

    i=0
    for batch in inferDataset:
        imageS1, imageS2, fileName = batch

        predS1, predTruckS1 = model([imageS1], training=False)
        predS2, predTruckS2 = model([imageS2], training=False)
        # print("fileName: ", fileName[-10:-1])
        # print("predS1 shape: ", predS1.shape)
        # print("predTruckS1 shape: ", predTruckS1.shape)
        # print("predS2 shape: ", predS2.shape)
        finalPred, attMask = attModel([predTruckS1, predS1, predS2], training = False)
        image = np.squeeze(finalPred)
        
        currentLogOdd = 0.8*np.log(image/(1-image)) + 0.2*previusLogOdd 
        
        previusLogOdd = np.copy(currentLogOdd)
        # for k in range(args.batch_size):
        # image = finalPred[k,:,:,:]
        labelArray = np.zeros((currentLogOdd.shape[0], currentLogOdd.shape[1]), dtype=np.uint8)
        labelArray[:,:] = np.argmax(currentLogOdd, axis=-1)
        # print("image: " , image.shape)
        # print("labelArray: " , labelArray.shape)

        label = Image.fromarray(labelArray)
        label.save(args.dataset_save_infer_path + "label_bayes/" + str(i).zfill(6)+".png")
        
        colorLabelArray = np.zeros((currentLogOdd.shape[0], currentLogOdd.shape[1], 3), dtype=np.uint8)
        colorLabelArray[:,:,:] = lb.cityscapes_pallete_float[np.argmax(currentLogOdd, axis=-1), :]*255
        # print("colorLabelArray: " , colorLabelArray.shape)
        
        colorLabel = Image.fromarray(colorLabelArray)
        colorLabel.save(args.dataset_save_infer_path + "color_bayes/" + str(i).zfill(6)+".png")
        
        i+=1
        
        print('\rbatch {}/{} {}{}'.format(int(i/args.batch_size), len(inferDataset), '>'*int(i/(30*args.batch_size)), '-'*int((len(inferDataset)-i/args.batch_size)/30)), end='', flush=True)

    return


def main():

    for arg in vars(args):
        print (arg, getattr(args, arg))
    
    # model = LMM()
    dataset = dl.dataloader(args)

    if(args.load_model_path == ''):
        model = resNet50V2Model(args.pre_train_model_path)
    else:
        print('Using trained model')
        model = keras.models.load_model(args.load_model_path, compile=False)

    # load data
    # trainDataset, valDataset, _, sizes = dl.loadDataset(args)

    # model.summary()

    if(args.mode == 'train'):
        print('Training network truck')
        trainDataset, valDataset, _, sizes = dataset.loadDataset()
        predictRand(model, 'train', index=10, filename='beforeTrain.png')
        print(len(trainDataset))
        trainTruck(model, trainDataset, valDataset, sizes)
        predictRand(model, 'train', index=10, filename='afterTrain.png')
    
    elif(args.mode == 'att'):
        print('Training network attention')
        trainDataset, valDataset, _, sizes = dataset.loadDataset()
        attModel = createAttModel(args.load_att_path)
        predictAttention(model, attModel, 'val', index=1)
        trainAtt(model, attModel, trainDataset, valDataset, sizes)
        predictAttention(model, attModel, 'val', filename='afterTrainAtt')

    elif(args.mode == 'eval_truck'):
        print('infering example')
        predictRand(model, 'train', index=10, filename='eval_truck2')

    elif(args.mode == 'eval_att'):
        print('infering att example')
        attModel = createAttModel(args.load_att_path)
        predictAttention(model, attModel, 'val', index=1, filename='testAttNewModel')
    
    elif(args.mode == 'inf_dataset'):
        attModel = createAttModel(args.load_att_path, args.img_height, args.img_width)
        attModel.save("att1V2.h5")
        print('infering folder: ', args.dataset_images_path)
        inferDataset = dataset.loadInferDataset()
        # attModel = createAttModel(args.load_att_path)
        inferData(model, attModel, inferDataset)
    else:
        print(args.mode, " not supported")
        return
    # if(args.save_model_path != ""):
    #     print("Saving model in " + args.save_model_path + "...")
    #     model.save(args.save_model_path)
    
    return
    

if __name__ == "__main__":
    main()
