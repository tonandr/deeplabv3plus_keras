"""
MIT License

Copyright (c) 2019 Inwoo Chung (gutomitai@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import argparse
import time
import platform
import json
import warnings
import shutil

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.io import imread, imsave
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dropout, DepthwiseConv2D,\
 Concatenate, Lambda, Activation, AveragePooling2D, SeparableConv2D
from tensorflow.keras.utils import multi_gpu_model
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite #?
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Constants.
DEBUG = True
MLFLOW_USAGE = False
NUM_CLASSES = 21

def get_one_hot(inputs, num_classes):
    """Get one hot tensor.
    
    Parameters
    ----------
    inputs: 3d numpy array (a x b x 1) 
        Input array.
    num_classes: integer
        Number of classes.
    
    Returns
    -------
    One hot tensor.
        3d numpy array (a x b x n).
    """
    onehots = np.zeros(shape=tuple(list(inputs.shape[:-1]) + [num_classes]))
    
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            try:
                onehots[i, j, inputs[i, j, 0]] = 1.0
            except IndexError:
                onehots[i, j, 0] = 1.0
        
    return onehots

MODE_TRAIN = 0
MODE_VAL = 1

class SemanticSegmentation(object):
    """Keras Semantic segmentation model of DeeplabV3+"""
    
    # Constants.
    #MODEL_PATH = 'semantic_segmentation_deeplabv3plus'
    MODEL_PATH = 'semantic_segmentation_deeplabv3plus.h5'
    #MODEL_PATH = 'semantic_segmentation_deeplabv3plus_is224_lr0_0001_ep344.h5'

    class TrainingSequencePascalVOC2012(Sequence):
        """Training data set sequence for Pascal VOC 2012."""
                
        def __init__(self, raw_data_path, hps, nn_arch, mode=MODE_TRAIN, eval=False):
            """
            Parameters
            ----------
            raw_data_path: string
                Raw data path.
            hps: dict
                Hyper-parameters.
            nn_arch: dict
                Model architecture.
            mode: string
                Training or validation mode.
            eval: boolean
                Evaluation flag.
            """
            self.raw_data_path = raw_data_path
            self.hps = hps
            self.nn_arch = nn_arch
            self.mode = mode
            self.eval = eval
            
            if self.mode == MODE_TRAIN:
                with open(os.path.join(self.raw_data_path
                                       , 'pascal-voc-2012'
                                       , 'VOC2012'
                                       , 'ImageSets'
                                       , 'Segmentation'
                                       , 'train.txt')) as f:
                    self.file_names = f.readlines() #?
            else:
                with open(os.path.join(self.raw_data_path
                                       , 'pascal-voc-2012'
                                       , 'VOC2012'
                                       , 'ImageSets'
                                       , 'Segmentation'
                                       , 'val.txt')) as f:
                    self.file_names = f.readlines() #?                
            
            # Remove \n.
            for i in range(len(self.file_names)):
                self.file_names[i] = self.file_names[i][:-1]
                
            self.total_samples = len(self.file_names)
            
            self.image_dir_path = os.path.join(self.raw_data_path
                                           , 'pascal-voc-2012'
                                           , 'VOC2012'
                                           , 'JPEGImages')
            self.label_dir_path = os.path.join(self.raw_data_path
                                           , 'pascal-voc-2012'
                                           , 'VOC2012'
                                           , 'SegmentationClass')
            
            self.batch_size = self.hps['batch_size']
            self.hps['step'] = self.total_samples // self.batch_size
            
            if self.total_samples % self.batch_size != 0:
                self.hps['temp_step'] = self.hps['step'] + 1
            else:
                self.hps['temp_step'] = self.hps['step']
                
        def __len__(self):
            return self.hps['temp_step']
        
        def __getitem__(self, index):
            images = []
            labels = []
            
            # Check the last index.
            if index == (self.hps['temp_step'] - 1):
                for bi in range(index * self.batch_size, len(self.file_names)):
                    file_name = self.file_names[bi] 
                    #if DEBUG: print(file_name )
                    
                    image_path = os.path.join(self.image_dir_path, file_name + '.jpg')
                    label_path = os.path.join(self.label_dir_path, file_name + '.png') #?
                    
                    # Load image.
                    image = imread(image_path)
                    image = 2.0 * (image / 255 - 0.5) # Normalization to (-1, 1).
                                                             
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    w = image.shape[1]
                    h = image.shape[0]
                    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                                    
                    if w >= h:
                        w_p = self.nn_arch['image_size']
                        h_p = int(h / w * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - h_p
                        
                        if pad % 2 == 0:
                            pad_t = pad // 2
                            pad_b = pad // 2
                        else:
                            pad_t = pad // 2
                            pad_b = pad // 2 + 1
        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                    else:
                        h_p = self.nn_arch['image_size']
                        w_p = int(w / h * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - w_p
                        
                        if pad % 2 == 0:
                            pad_l = pad // 2
                            pad_r = pad // 2
                        else:
                            pad_l = pad // 2
                            pad_r = pad // 2 + 1                
                        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                        
                    # Load label.
                    label = np.expand_dims(np.array(Image.open(label_path)), axis=-1)
                    label[label > (self.nn_arch['num_classes'] - 1)] = 0
                                                             
                    # Adjust the original label size into the normalized label size according to the ratio of width, height.
                    w = label.shape[1]
                    h = label.shape[0]
                    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                                    
                    if w >= h:
                        w_p = self.nn_arch['image_size']
                        h_p = int(h / w * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - h_p
                        
                        if pad % 2 == 0:
                            pad_t = pad // 2
                            pad_b = pad // 2
                        else:
                            pad_t = pad // 2
                            pad_b = pad // 2 + 1
        
                        label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        label = cv.copyMakeBorder(label, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                    else:
                        h_p = self.nn_arch['image_size']
                        w_p = int(w / h * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - w_p
                        
                        if pad % 2 == 0:
                            pad_l = pad // 2
                            pad_r = pad // 2
                        else:
                            pad_l = pad // 2
                            pad_r = pad // 2 + 1                
                        
                        label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        label = cv.copyMakeBorder(label, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                    
                    # Convert label to one hot label.
                    label = np.expand_dims(label, axis=-1)
                    label[label > (self.nn_arch['num_classes'] - 1)] = 0
                    if self.mode == MODE_TRAIN and self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        
                    images.append(image)
                    labels.append(label)          
            else:
                for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                    file_name = self.file_names[bi] 
                    #if DEBUG: print(file_name )
                    
                    image_path = os.path.join(self.image_dir_path, file_name + '.jpg')
                    label_path = os.path.join(self.label_dir_path, file_name + '.png') #?
                    
                    # Load image.
                    image = imread(image_path)
                    image = 2.0 * (image / 255 - 0.5) # Normalization to (-1, 1).
                                                             
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    w = image.shape[1]
                    h = image.shape[0]
                    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                                    
                    if w >= h:
                        w_p = self.nn_arch['image_size']
                        h_p = int(h / w * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - h_p
                        
                        if pad % 2 == 0:
                            pad_t = pad // 2
                            pad_b = pad // 2
                        else:
                            pad_t = pad // 2
                            pad_b = pad // 2 + 1
        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                    else:
                        h_p = self.nn_arch['image_size']
                        w_p = int(w / h * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - w_p
                        
                        if pad % 2 == 0:
                            pad_l = pad // 2
                            pad_r = pad // 2
                        else:
                            pad_l = pad // 2
                            pad_r = pad // 2 + 1                
                        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                        
                    # Load label.
                    label = np.expand_dims(np.array(Image.open(label_path)), axis=-1)
                    label[label > (self.nn_arch['num_classes'] - 1)] = 0
                                                             
                    # Adjust the original label size into the normalized label size according to the ratio of width, height.
                    w = label.shape[1]
                    h = label.shape[0]
                    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                                    
                    if w >= h:
                        w_p = self.nn_arch['image_size']
                        h_p = int(h / w * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - h_p
                        
                        if pad % 2 == 0:
                            pad_t = pad // 2
                            pad_b = pad // 2
                        else:
                            pad_t = pad // 2
                            pad_b = pad // 2 + 1
        
                        label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        label = cv.copyMakeBorder(label, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                    else:
                        h_p = self.nn_arch['image_size']
                        w_p = int(w / h * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - w_p
                        
                        if pad % 2 == 0:
                            pad_l = pad // 2
                            pad_r = pad // 2
                        else:
                            pad_l = pad // 2
                            pad_r = pad // 2 + 1                
                        
                        label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        label = cv.copyMakeBorder(label, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                    
                    # Convert label to one hot label.
                    label = np.expand_dims(label, axis=-1)
                    label[label > (self.nn_arch['num_classes'] - 1)] = 0
                    if self.mode == MODE_TRAIN and self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                    
                    images.append(image)
                    labels.append(label)
                                                                         
            return (np.asarray(images), np.asarray(labels))

    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: dictionary
            Semantic segmentation model configuration dictionary.
        """
        
        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        self.model_loading = self.conf['model_loading']
                
        if self.model_loading:
            opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay']) 
            with CustomObjectScope({}): 
                if self.conf['multi_gpu']:
                    self.model = load_model(self.MODEL_PATH)
                    
                    self.parallel_model = multi_gpu_model(self.model, gpus=self.conf['num_gpus'])
                    self.parallel_model.compile(optimizer=opt
                                                , loss=self.model.losses
                                                , metrics=self.model.metrics)
                else:
                    self.model = load_model(os.path.join(self.raw_data_path, self.MODEL_PATH))
                    #self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)])
        else:
            # Design the semantic segmentation model.
            # Load mobilenetv2 as the base model.
            mv2 = MobileNetV2(include_top=False) #, depth_multiplier=self.nn_arch['mv2_depth_multiplier'])
            self.base = Model(inputs=mv2.inputs
                              , outputs=mv2.get_layer('block_5_add').output) # Layer satisfying output stride of 8.
            self.base.trainable = False
            for layer in self.base.layers: layer.trainable = False #?
            
            self.base._init_set_name('base') 
            
            # Make the encoder-decoder model.
            self._make_encoder()
            self._make_decoder()
            
            inputs = self.encoder.inputs
            features = self.encoder(inputs)
            #outputs = self.decoder([inputs[0], features])
            outputs = self.decoder(features)
            
            self.model = Model(inputs, outputs)
            
            # Compile.
            opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay'])
            
            self.model.compile(optimizer=opt
                               , loss='categorical_crossentropy')
                               #, metrics=[tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)])
            self.model._init_set_name('deeplabv3plus_mnv2')
            
            if self.conf['multi_gpu']:
                self.parallel_model = multi_gpu_model(self.model, gpus=self.conf['num_gpus'])
                self.parallel_model.compile(optimizer=opt
                                            , loss=self.model.losses
                                            , metrics=self.model.metrics)
            
    def _make_encoder(self):
        """Make encoder."""
        assert hasattr(self, 'base')
        
        # Inputs.
        input_image = Input(shape=(self.nn_arch['image_size']
                                       , self.nn_arch['image_size']
                                       , 3)
                                       , name='input_image')
        
        # Extract feature.
        x = self.base(input_image)
        
        # Conduct dilated convolution pooling.
        pooled_outputs = []
        for conf in self.nn_arch["encoder_middle_conf"]:
            if conf['input'] == -1:
                x2 = x #?
            else:
                x2 = pooled_outputs[conf['input']]
            
            if conf['op'] == 'conv': 
                if conf['kernel'] == 1:
                    x2 = Conv2D(self.nn_arch['reduction_size']
                                , kernel_size=1
                                , padding='same'
                                , use_bias=False
                                , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x2)
                    x2 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(x2)
                    x2 = Activation('relu')(x2)                
                else:
                    # Split separable conv2d.
                    x2 = SeparableConv2D(self.nn_arch['reduction_size'] #?
                                        , conf['kernel']
                                        , depth_multiplier=1
                                        , dilation_rate=(conf['rate'][0] * self.nn_arch['conv_rate_multiplier']
                                                         , conf['rate'][1] * self.nn_arch['conv_rate_multiplier'])
                                        , padding='same'
                                        , use_bias=False
                                        , kernel_initializer=initializers.TruncatedNormal())(x2)
                    x2 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(x2)
                    x2 = Activation('relu')(x2) 
                    x2 = Conv2D(self.nn_arch['reduction_size']
                                , kernel_size=1
                                , padding='same'
                                , use_bias=False
                                , kernel_initializer=initializers.TruncatedNormal()
                                , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x2)
                    x2 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(x2)
                    x2 = Activation('relu')(x2) 
            elif conf['op'] == 'pyramid_pooling':
                x2 = AveragePooling2D(pool_size=conf['kernel'], padding='valid')(x2)
                x2 = Conv2D(self.nn_arch['reduction_size']
                            , kernel_size=1
                            , padding='same'
                            , use_bias=False
                            , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x2)
                x2 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(x2)
                x2 = Activation('relu')(x2) 
                
                target_size = conf['target_size_factor'] #?
                x2 = Lambda(lambda x: K.resize_images(x
                                             , target_size[0]
                                             , target_size[1]
                                             , "channels_last"
                                             , interpolation='bilinear'))(x2) #?
            else:
                raise ValueError('Invalid operation.')
            
            pooled_outputs.append(x2)
                
        # Concatenate pooled tensors.
        x3 = Concatenate(axis=-1)(pooled_outputs)
        x3 = Dropout(rate=self.nn_arch['dropout_rate'])(x3)
        x3 = Conv2D(self.nn_arch['concat_channels']
                    , kernel_size=1
                    , padding='same'
                    , use_bias=False
                    , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x3)
        x3 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(x3)
        x3 = Activation('relu')(x3) 
        output = Dropout(rate=self.nn_arch['dropout_rate'])(x3)
        
        self.encoder = Model(input_image, output)
        self.encoder._init_set_name('encoder')
        
    def _make_decoder(self):
        """Make decoder."""
        assert hasattr(self, 'base') and hasattr(self, 'encoder')
        
        inputs = self.encoder.outputs
        features = Input(shape=K.int_shape(inputs[0])[1:])
        x = features 
        
        # Refine boundary.
        #low_features = Input(shape=K.int_shape(self.encoder.inputs[0])[1:])
        #x = self._refine_boundary(low_features, features)
        
        # Upsampling & softmax.
        x = Conv2D(self.nn_arch['num_classes']
                   , kernel_size=3
                   , padding='same'
                   , use_bias=False
                   , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x) # Kernel size?
        
        output_stride = self.nn_arch['output_stride']
        x = Lambda(lambda x: K.resize_images(x
                                             , output_stride #4
                                             , output_stride #4
                                             , "channels_last"
                                             , interpolation='bilinear'))(x) #?
        outputs = Activation('softmax')(x)
        
        #self.decoder = Model(inputs=[low_features, features], outputs=outputs)
        self.decoder = Model(inputs=[features], outputs=outputs)
        self.decoder._init_set_name('decoder')
    
    def _refine_boundary(self, low_features, features):
        """Refine segmentation boundary.
        
        Parameters
        ----------
        low_features: Tensor
            Image input tensor.
        features: Tensor
            Encoder's output tensor.
        
        Returns
        -------
        Refined features.
            Tensor
        """
        low_features = self.base(low_features)
        low_features = Conv2D(48
                        , kernel_size=1
                        , padding='same'
                        , use_bias=False
                        , kernel_regularizer=regularizers.l2(self.hps['weight_decay'])
                        , activation='relu')(low_features)
        low_features = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(low_features)
        
        # Resize low_features, features.
        output_stride = self.nn_arch['output_stride']       
        low_features = Lambda(lambda x: K.resize_images(x
                                             , output_stride / 2
                                             , output_stride / 2
                                             , "channels_last"
                                             , interpolation='bilinear'))(low_features) #?
        features = Lambda(lambda x: K.resize_images(x
                                             , output_stride / 2
                                             , output_stride / 2
                                             , "channels_last"
                                             , interpolation='bilinear'))(features) #?
        
        x = Concatenate(axis=-1)([low_features, features])
                 
        return x 
       
    def train(self):
        """Train."""        
        trGen = self.TrainingSequencePascalVOC2012(self.raw_data_path, self.hps, self.nn_arch, mode=MODE_TRAIN)
        assert 'step' in self.hps.keys()
        
        reduce_lr = ReduceLROnPlateau(monitor='loss'
                                      , factor=self.hps['reduce_lr_factor']
                                      , patience=5
                                      , min_lr=0.00000001
                                      , verbose=1)
        model_check_point = ModelCheckpoint(os.path.join(self.raw_data_path, self.MODEL_PATH)
                                            , monitor='loss'
                                            , verbose=1
                                            , save_best_only=True)
        
        '''
        def schedule_lr(e_i):
            self.hps['lr'] = self.hps['reduce_lr_factor'] * self.hps['lr']
            return self.hps['lr']
        
        lr_scheduler = LearningRateScheduler(schedule_lr, verbose=1)
        '''
        
        if self.conf['multi_gpu']:
            self.parallel_model.fit_generator(trGen
                          , steps_per_epoch=self.hps['step'] #?                   
                          , epochs=self.hps['epochs']
                          , verbose=1
                          , max_queue_size=80
                          , workers=8
                          , use_multiprocessing=False
                          , callbacks=[model_check_point, reduce_lr])
        else:     
            self.model.fit_generator(trGen
                          , steps_per_epoch=self.hps['step']                  
                          , epochs=self.hps['epochs']
                          , verbose=1
                          , max_queue_size=100
                          , workers=1
                          , use_multiprocessing=False
                          , callbacks=[model_check_point, reduce_lr])

        print('Save the model.')
        self.model.save(os.path.join(self.raw_data_path, self.MODEL_PATH), save_format='h5')            
        #self.model.save(os.path.join(self.raw_data_path, self.MODEL_PATH), save_format='tf')
        
    def evaluate(self):
        """Evaluate.
        
        Returns
        -------
        Mean iou.
            Scalar float.
        """
        assert hasattr(self, 'model')

        # Initialize the results directory
        if not os.path.isdir(os.path.join(self.raw_data_path, 'results')):
            os.mkdir(os.path.join(self.raw_data_path, 'results'))
        else:
            shutil.rmtree(os.path.join(self.raw_data_path, 'results'))
            os.mkdir(os.path.join(self.raw_data_path, 'results'))

        valGen = self.TrainingSequencePascalVOC2012(self.raw_data_path
                                                    , self.hps
                                                    , self.nn_arch
                                                    , mode=MODE_TRAIN
                                                    , eval=True)
        use_multiprocessing = False
        max_queue_size = 80
        workers = 4
        shuffle = False
                   
        # Check exception.
        if not isinstance(valGen, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multi processing, use the instance of Sequence.'))
        
        try:        
            # Get the output generator.
            if workers > 0:
                if isinstance(valGen, Sequence):
                    enq = OrderedEnqueuer(valGen
                                      , use_multiprocessing=use_multiprocessing
                                      , shuffle=shuffle)
                else:
                    enq = GeneratorEnqueuer(valGen
                                            , use_multiprocessing=use_multiprocessing)
                    
                enq.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enq.get()
            else:
                if isinstance(valGen, Sequence):
                    output_generator = iter_sequence_infinite(valGen)
                else:
                    output_generator = valGen
            
            c_miou = tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)
            pbar = tqdm(range(self.hps['step']))                                    
            for s_i in pbar: #?
                images, labels = next(output_generator)
                labels[labels > (self.nn_arch['num_classes'] - 1)] = 0
                
                results = self.segment(images)
                results[results > (self.nn_arch['num_classes'] - 1)] = 0
                                   
                c_miou.update_state(results, labels)
                pbar.set_description("Mean IOU: {}".format(c_miou.result().numpy()))
                
                # Save result images.
                for i in range(self.hps['batch_size']):
                    plt.subplot(121); plt.imshow(np.squeeze(labels[i]))
                    plt.subplot(122); plt.imshow(np.squeeze(results[i]))
                    plt.savefig(os.path.join(self.raw_data_path
                                             , 'results'
                                             , 'result_{0:d}.png'.format(s_i * self.hps['batch_size'] + i)))                
        finally:
            try:
                if enq is not None:
                    enq.stop()
            finally:
                pass
        
        print('Mean iou: {}'.format(c_miou))
        return c_miou
            
    def segment(self, images):
        """Segment semantic regions. 
        
        Parameters
        ----------
        images: 4d numpy array
            Images.
        
        Returns
        -------
        Segmented result labels.
            4d numpy array.
        """
        assert hasattr(self, 'model')
        
        if self.conf['multi_gpu']:
            onehots = self.parallel_model.predict(images) #?
            labels = np.zeros(shape=tuple(list(onehots.shape[:-1]) + [1]), dtype=np.uint8)
            
            for b_i in range(onehots.shape[0]):
                for h_i in range(onehots.shape[1]):
                    for w_i in range(onehots.shape[2]):
                        labels[b_i, h_i, w_i, 0] = np.argmax(onehots[b_i, h_i, w_i, :])
        else:
            onehots = self.model.predict(images)
            labels = np.zeros(shape=tuple(list(onehots.shape[:-1]) + [1]), dtype=np.uint8)
            
            for b_i in range(onehots.shape[0]):
                for h_i in range(onehots.shape[1]):
                    for w_i in range(onehots.shape[2]):
                        labels[b_i, h_i, w_i, 0] = np.argmax(onehots[b_i, h_i, w_i, :])
                        
        return labels 
                          
def main():
    """Main."""

    # Load configuration.
    if platform.system() == 'Windows':
        with open("semantic_segmentation_deeplabv3plus_conf_win.json", 'r') as f:
            conf = json.load(f)   
    else:
        with open("semantic_segmentation_deeplabv3plus_conf.json", 'r') as f:
            conf = json.load(f)   
    
    if conf['mode'] == 'train':        
        # Train.
        ss = SemanticSegmentation(conf)
        
        ts = time.time()
        ss.train()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif conf['mode'] == 'evaluate':
        # Evaluate.
        ss = SemanticSegmentation(conf)
        
        ts = time.time()
        ss.evaluate()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))                   
             
if __name__ == '__main__':
    main()
