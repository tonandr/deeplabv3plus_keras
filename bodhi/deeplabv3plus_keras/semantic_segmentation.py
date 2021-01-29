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

#@PydevCodeAnalysisIgnore

import os
import argparse
import time
import platform
import json
import warnings
import shutil
import random

import numpy as np
import cv2 as cv
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0
from tqdm import tqdm
import pandas as pd
from scipy import ndimage
from cupyx.scipy import ndimage as ndimage_gpu
import numpy as np
import cupy as cp

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dropout
from tensorflow.keras.layers import (Concatenate
    , Lambda
    , Activation
    , AveragePooling2D
    , SeparableConv2D)
from tensorflow.keras import optimizers
from tensorflow.keras.applications import MobileNetV2, Xception
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.metrics import MeanIoU
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops, array_ops, confusion_matrix

from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite

#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Constants.
DEBUG = True

DEVICE_CPU = -1

MODE_TRAIN = 0
MODE_VAL = 1
MODE_TEST = 2

BASE_MODEL_MOBILENETV2 = 'mobilenetv2'
BASE_MODEL_XCEPTION = 'xception'
BASE_MODEL_EFFICIENTNETB0 = 'efficientnetb0'

RESOURCE_TYPE_PASCAL_VOC_2012 = 'pascal_voc_2012'
RESOURCE_TYPE_PASCAL_VOC_2012_EXT = 'pascal_voc_2012_ext'
RESOURCE_TYPE_GOOGLE_OPEN_IMAGES_V5 = 'google_open_images_v5'

GOIV5_SPECIFIC_SET = set(['Person', 'Cat', 'Dog', 'Car', 'Bus', 'Motorcycle', 'Bicyle'])

ss_pw = [0.29754999, 0.99106889, 0.99236374, 0.99122957, 0.99350396, 0.99455487,
 0.98728424, 0.98090446, 0.96883489, 0.98753125, 0.99376389, 0.98942612,
 0.97222875, 0.99080578, 0.98845309, 0.92606652, 0.99393374, 0.99374322,
 0.98782171, 0.98659656, 0.99233476]
ss_nw = [0.70245001, 0.00893111, 0.00763626, 0.00877043, 0.00649604, 0.00544513,
 0.01271576, 0.01909554, 0.03116511, 0.01246875, 0.00623611, 0.01057388,
 0.02777125, 0.00919422, 0.01154691, 0.07393348, 0.00606626, 0.00625678,
 0.01217829, 0.01340344, 0.00766524]


def resize(image, size: tuple, mode='constant', device: int = DEVICE_CPU):
    '''Resize image using scipy's affine_transform.

    Parameters
    ----------
    image: 3d numpy array or cypy array.
        Image data.
    size: Tuple.
        Target width and height.
    mode: String.
        Boundary mode (default is constant).
    device: Integer.
        Device kind (default is cpu).

    Returns
    -------
    3d numpy or cypy array.
        Resized image data.
    '''

    # Calculate x, y scaling factors and output shape.
    w, h = size
    h_o, w_o, _ = image.shape
    fx = w / float(w_o)
    fy = h / float(h_o)
    output_shape = (h, w, image.shape[2])

    # Calculate resizing according to device.
    if device == DEVICE_CPU:
        # Create affine transformation matrix.
        M = np.eye(4)
        M[0, 0] = 1.0 / fy
        M[1, 1] = 1.0 / fx
        M = M[0:3]

        # Resize.
        resized_image = ndimage.affine_transform(image
                                                 , M
                                                 , order=1
                                                 , output_shape=output_shape
                                                 , mode=mode)

        return resized_image
    elif device >= DEVICE_CPU:
        if hasattr(image, 'device') != True:
            image_gpu = cp.asarray(image)
        else:
            image_gpu = image

        # Create affine transformation matrix.
        M_gpu = cp.eye(4)
        M_gpu[0, 0] = 1.0 / fy
        M_gpu[1, 1] = 1.0 / fx
        M_gpu = M_gpu[0:3]

        # Resize.
        resized_image = ndimage_gpu.affine_transform(image_gpu
                                                     , M_gpu
                                                     , order=1
                                                     , output_shape=output_shape
                                                     , mode=mode)

        if hasattr(image, 'device') != True:
            return resized_image.get()
        else:
            return resized_image
    else:
        raise ValueError('device is not valid.')


def resize_image_to_target_symmeric_size(image, size: int, device=DEVICE_CPU):
    """Resize image to target symmetric size.

    Parameters
    ----------
    image: 3d numpy array or cypy array.
        Image data.
    size: Integer.
        Target symmetric image size.
    device: Integer.
        Device kind (default is cpu).

    Returns
    -------
    3d numpy or cypy array.
        Resized image data.
    """

    # Adjust the original image size into the normalized image size according to the ratio of width, height.
    w = image.shape[1]
    h = image.shape[0]
    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0

    if w >= h:
        w_p = size
        h_p = int(h / w * size)
        pad = size - h_p

        if pad % 2 == 0:
            pad_t = pad // 2
            pad_b = pad // 2
        else:
            pad_t = pad // 2
            pad_b = pad // 2 + 1

        image_p = resize(image, (w_p, h_p), mode='nearest', device=device)

        if device == DEVICE_CPU:
            image_p = np.pad(image_p, ((pad_t, pad_b), (0, 0), (0, 0)))
        elif device > DEVICE_CPU:
            if hasattr(image, 'device') != True:
                image_gpu = cp.asarray(image_p)
            else:
                image_gpu = image_p

            image_gpu = cp.pad(image_gpu, ((pad_t, pad_b), (0, 0), (0, 0)))

            if hasattr(image, 'device') != True:  # ?
                image_p = image_gpu.get()
            else:
                image_p = image_gpu
    else:
        h_p = size
        w_p = int(w / h * size)
        pad = size - w_p

        if pad % 2 == 0:
            pad_l = pad // 2
            pad_r = pad // 2
        else:
            pad_l = pad // 2
            pad_r = pad // 2 + 1

        image_p = resize(image, (w_p, h_p), mode='nearest', device=device)

        if device == DEVICE_CPU:
            image_p = np.pad(image_p, ((0, 0), (pad_r, pad_l), (0, 0)))
        elif device > DEVICE_CPU:
            if hasattr(image, 'device') != True:
                image_gpu = cp.asarray(image_p)
            else:
                image_gpu = image_p

            image_gpu = cp.pad(image_gpu, ((0, 0), (pad_r, pad_l), (0, 0)))

            if hasattr(image, 'device') != True:  # ?
                image_p = image_gpu.get()
            else:
                image_p = image_gpu

    return image_p, w, h, pad_t, pad_l, pad_b, pad_r


class MeanIoUExt(MeanIoU):
    """Calculate the mean IoU for one hot truth and prediction vectors."""

    def __init__(self, num_classes, accum_enable=True, name=None, dtype=None):
        super(MeanIoUExt, self).__init__(num_classes, name=name, dtype=dtype)
        self.accum_enable = accum_enable

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulated the confusion matrix statistics with one hot truth and prediction data.

        Parameters
        ----------
        y_true: Tensor or numpy array.
            One hot ground truth vectors.
        y_pred: Tensor or numpy array.
            One hot predicted vectors.
        sample_weight: Tensor.
            Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns
        -------
        Update operator.
            Operator
        """
        # Convert one hot vectors to labels.
        y_true = K.argmax(y_true)
        y_pred = K.argmax(y_pred)

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = array_ops.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = array_ops.reshape(y_true, [-1])

        if sample_weight is not None and sample_weight.shape.ndims > 1:
            sample_weight = array_ops.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=dtypes.float64)
        return self.total_cm.assign_add(current_cm) if self.accum_enable \
            else self.total_cm.assign(current_cm)


def get_one_hot(label, num_classes): #?
    """Get one hot tensor.

    Parameters
    ----------
    label: Numpy array.
        label.
    num_classes: Integer
        Number of classes.

    Returns
    -------
    One hot.
        Numpy array.
    """
    indexes = label.ravel()
    shape = tuple(list(label.shape[:-1]) + [num_classes])
    onehot = np.zeros(shape=shape)
    onehot = onehot.ravel()

    for i in range(label.size):
        onehot[i * num_classes + indexes[i]] = 1

    onehot = onehot.reshape(shape)

    return onehot


def cal_ss_class_imbalance_weights(resource_path, size=21):
    with open(os.path.join(resource_path
            , 'VOCdevkit'
            , 'VOC2012'
            , 'ImageSets'
            , 'Segmentation'
            , 'train_aug_val.txt')) as f:
        file_names = f.readlines()  # ?

    # Remove \n.
    for i in range(len(file_names)):
        file_names[i] = file_names[i][:-1]

    label_dir_path = os.path.join(resource_path
                                  , 'VOCdevkit'
                                  , 'VOC2012'
                                  , 'SegmentationClassAug')

    pf = np.zeros(size)
    total_num = 0.0

    for i in tqdm(range(len(file_names))):
        file_name = file_names[i]

        # Load label.
        label_path = os.path.join(label_dir_path, file_name + '.png')  # ?

        label = imread(label_path)
        label[label > (size - 1)] = 0

        label_oh = get_one_hot(label, size)
        label2 = label_oh.reshape(np.prod(label.shape), size)
        label_pf = label2.sum(axis=0)
        pf = pf + label_pf
        total_num += np.prod(label.shape)

    pf = pf / total_num
    nf = 1.0 - pf
    pw = nf
    nw = pf

    print(f'pw: {pw}, nw: {nw}')
    return pw, nw


def class_imbalance_loss(pos_weights, neg_weights, epsilon=1e-7):
    def loss_f(y_true, y_pred):
        # initialize loss to zero
        loss = 0.0

        for i in range(len(pos_weights)):
            loss += -1.0 * (pos_weights[i] * y_true[..., i] * K.log(y_pred[..., i] + epsilon)
                            + neg_weights[i] * (1.0 - y_true[..., i]) * K.log(
                        1.0 - y_pred[..., i] + epsilon))
        return K.mean(loss)
    return loss_f


class ClassBalancedLoss(LossFunctionWrapper):
    def __init__(self
                 , pos_weights = 1.0
                 , neg_weights = 0.0
                 , epsilon=1e-7
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='class_balanced_loss'):
        super(ClassBalancedLoss, self).__init__(class_balanced_loss
            , pos_weights=pos_weights
            , neg_weights=neg_weights
            , epsilon=epsilon
            , reduction=reduction
            , name=name)


@tf.autograph.experimental.do_not_convert
def class_balanced_loss(y_true, y_pred, pos_weights=1.0, neg_weights=0.0, epsilon=1e-7):
        # initialize loss to zero
        loss = 0.0

        for i in range(len(pos_weights)):
            loss += -1.0 * (pos_weights[i] * y_true[..., i] * K.log(y_pred[..., i] + epsilon)
                            + neg_weights[i] * (1.0 - y_true[..., i]) * K.log(
                        1.0 - y_pred[..., i] + epsilon))
        return K.mean(loss)


class SemanticSegmentation(object):
    """Keras Semantic segmentation model of DeeplabV3+"""
    
    # Constants.
    MODEL_PATH = 'semantic_segmentation_deeplabv3plus'
    #MODEL_PATH = 'semantic_segmentation_deeplabv3plus.h5'
    TF_LITE_MODEL_PATH = 'semantic_segmentation_deeplabv3plus.tflite'
    #MODEL_PATH = 'semantic_segmentation_deeplabv3plus_is224_lr0_0001_ep344.h5'

    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: Dictionary.
            Semantic segmentation model configuration dictionary.
        """
        
        # Check exception.
        assert conf['nn_arch']['output_stride'] == 8 or conf['nn_arch']['output_stride'] == 16 
        
        # Initialize.
        self.conf = conf
        self.resource_path = self.conf['resource_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        self.model_loading = self.conf['model_loading']

        opt = optimizers.Adam(lr=self.hps['lr']
                              , beta_1=self.hps['beta_1']
                              , beta_2=self.hps['beta_2']
                              , decay=self.hps['decay'])

        if self.model_loading:
            with CustomObjectScope({'ClassBalancedLoss': ClassBalancedLoss
                                    , 'MeanIoUExt': MeanIoUExt}):
                self.model = load_model(os.path.join(self.resource_path, self.MODEL_PATH))
                '''
                self.model.compile(optimizer=opt
                                   , loss=ClassBalancedLoss(ss_pw, ss_nw)
                                   , metrics=[MeanIoUExt(num_classes=self.nn_arch['num_classes'])])
                '''
        else:
            # Design the semantic segmentation model.
            # Load a base model.
            if self.conf['base_model'] == BASE_MODEL_MOBILENETV2:
                # Load mobilenetv2 as the base model.
                mv2 = MobileNetV2(input_shape=(self.nn_arch['image_size']
                                       , self.nn_arch['image_size']
                                       , 3)
                                    , include_top=False) #, depth_multiplier=self.nn_arch['mv2_depth_multiplier'])
                
                if self.nn_arch['output_stride'] == 8:
                    self.base = Model(inputs=mv2.inputs, outputs=mv2.get_layer('block_5_add').output) # Layer satisfying output stride of 8.
                else:
                    self.base = Model(inputs=mv2.inputs, outputs=mv2.get_layer('block_12_add').output) # Layer satisfying output stride of 16.
                
                self.base.trainable = True
                for layer in self.base.layers: layer.trainable = True #?
                
                self.base._init_set_name('base')
            elif self.conf['base_model'] == BASE_MODEL_XCEPTION:
                # Load xception as the base model.
                xception = Xception(input_shape=(self.nn_arch['image_size']
                                       , self.nn_arch['image_size']
                                       , 3)
                                    , include_top=False) #, depth_multiplier=self.nn_arch['mv2_depth_multiplier'])
                
                if self.nn_arch['output_stride'] == 8:
                    self.base = Model(inputs=xception.inputs, outputs=xception.get_layer('block4_sepconv2_bn').output) # Layer satisfying output stride of 8.
                else:
                    self.base = Model(inputs=xception.inputs, outputs=xception.get_layer('block13_sepconv2_bn').output) # Layer satisfying output stride of 16.
                
                self.base.trainable = True
                for layer in self.base.layers: layer.trainable = True #?
                
                self.base._init_set_name('base')
            elif self.conf['base_model'] == BASE_MODEL_EFFICIENTNETB0:
                # Load efficientnetb0 as the base model.
                effnetb0 = EfficientNetB0(input_shape=(self.nn_arch['image_size']
                                            , self.nn_arch['image_size']
                                            , 3)
                               , include_top=False)  # , depth_multiplier=self.nn_arch['mv2_depth_multiplier'])

                if self.nn_arch['output_stride'] == 8:
                    self.base = Model(inputs=effnetb0.inputs, outputs=effnetb0.get_layer('block3b_add').output)  # Layer satisfying output stride of 8.
                else:
                    self.base = Model(inputs=effnetb0.inputs, outputs=effnetb0.get_layer('block5c_add').output)  # Layer satisfying output stride of 16.

                self.base.trainable = True
                for layer in self.base.layers: layer.trainable = True  # ?

                self.base._init_set_name('base')

                # Make the encoder-decoder model.
            self._make_encoder()
            self._make_decoder()
            
            inputs = self.encoder.inputs
            features = self.encoder(inputs)
            outputs = self.decoder([inputs[0], features]) if self.nn_arch['boundary_refinement'] \
                else self.decoder(features)
            
            self.model = Model(inputs, outputs)
            
            # Compile.
            self.model.compile(optimizer=opt
                               , loss=ClassBalancedLoss(ss_pw, ss_nw)
                               , metrics=[MeanIoUExt(num_classes=self.nn_arch['num_classes'])])
            self.model._init_set_name('deeplabv3plus')

    def _make_encoder(self):
        """Make encoder."""
        assert hasattr(self, 'base')
        
        # Inputs.
        input_image = Input(shape=(self.nn_arch['image_size']
                                       , self.nn_arch['image_size']
                                       , 3)
                                       , dtype=self.hps['dtype']
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
        #output = Dropout(rate=self.nn_arch['dropout_rate'])(x3)
        output = x3
        
        self.encoder = Model(input_image, output)
        self.encoder._init_set_name('encoder')
        
    def _make_decoder(self):
        """Make decoder."""
        assert hasattr(self, 'base') and hasattr(self, 'encoder')
        
        inputs = self.encoder.outputs
        features = Input(shape=K.int_shape(inputs[0])[1:], dtype=self.hps['dtype'])
 
        if self.nn_arch['boundary_refinement']:
            # Refine boundary.
            low_features = Input(shape=K.int_shape(self.encoder.inputs[0])[1:])
            x = self._refine_boundary(low_features, features)
        else:
            x = features
        
        # Upsampling & softmax.
        x = Conv2D(self.nn_arch['num_classes']
                   , kernel_size=3
                   , padding='same'
                   , use_bias=False
                   , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(x) # Kernel size?
        
        output_stride = self.nn_arch['output_stride']
        
        if self.nn_arch['boundary_refinement']:
            output_stride = int(output_stride / 8 if output_stride == 16 else output_stride / 4)
            
        x = Lambda(lambda x: K.resize_images(x
                                             , output_stride
                                             , output_stride
                                             , "channels_last"
                                             , interpolation='bilinear'))(x) #?
        outputs = Activation('softmax')(x)
        
        self.decoder = Model(inputs=[low_features, features], outputs=outputs) if self.nn_arch['boundary_refinement'] \
            else Model(inputs=[features], outputs=outputs)
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
                        , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))(low_features)
        low_features = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])(low_features)
        low_features = Activation('relu')(low_features) 
        
        # Resize low_features, features.
        output_stride = self.nn_arch['output_stride']       
        low_features = Lambda(lambda x: K.resize_images(x
                                             , int(output_stride / 2)
                                             , int(output_stride / 2)
                                             , "channels_last"
                                             , interpolation='bilinear'))(low_features) #?
        features = Lambda(lambda x: K.resize_images(x
                                             , int(output_stride / 2)
                                             , int(output_stride / 2)
                                             , "channels_last"
                                             , interpolation='bilinear'))(features) #?
        
        x = Concatenate(axis=-1)([low_features, features])
                 
        return x 
       
    def train(self):
        """Train.""" 
        if self.conf['resource_type'] == RESOURCE_TYPE_PASCAL_VOC_2012: 
            tr_gen = self.TrainingSequencePascalVOC2012(self.conf
                                                       , mode=MODE_TRAIN)
            val_gen = self.TrainingSequencePascalVOC2012(self.conf
                                                        , mode=MODE_VAL)
        elif self.conf['resource_type'] == RESOURCE_TYPE_PASCAL_VOC_2012_EXT:
            tr_gen = self.TrainingSequencePascalVOC2012Ext(self.conf
                                                       , mode=MODE_TRAIN)
            val_gen = self.TrainingSequencePascalVOC2012Ext(self.conf
                                                        , mode=MODE_VAL)
        elif self.conf['resource_type'] == RESOURCE_TYPE_GOOGLE_OPEN_IMAGES_V5:
            tr_gen = self.TrainingSequenceGoogleOpenImagesV5(self.conf
                                                       , mode=MODE_TRAIN)
            val_gen = self.TrainingSequenceGoogleOpenImagesV5(self.conf
                                                        , mode=MODE_VAL)
        else:
            raise ValueError('resource type is not valid.')
        
        assert 'tr_step' in self.hps.keys() and 'val_step' in self.hps.keys()
        
        reduce_lr = ReduceLROnPlateau(monitor='loss'
                                      , factor=self.hps['reduce_lr_factor']
                                      , patience=5
                                      , min_lr=1.e-8
                                      , verbose=1)
        model_check_point = ModelCheckpoint(os.path.join(self.resource_path, self.MODEL_PATH)
                                            , monitor='val_loss'
                                            , verbose=1
                                            , save_best_only=True)
        tensorboard = tf.keras.callbacks.TensorBoard(histogram_freq=1
               , write_graph=True
               , write_images=True
               , update_freq='epoch')
        
        '''
        def schedule_lr(e_i):
            self.hps['lr'] = self.hps['reduce_lr_factor'] * self.hps['lr']
            return self.hps['lr']
        
        lr_scheduler = LearningRateScheduler(schedule_lr, verbose=1)
        '''

        self.model.fit(tr_gen
                          , steps_per_epoch=self.hps['tr_step']                  
                          , epochs=self.hps['epochs']
                          , verbose=1
                          , max_queue_size=self.conf['max_queue_size']
                          , workers=self.conf['workers']
                          , use_multiprocessing=False
                          , callbacks=[model_check_point, reduce_lr] #, tensorboard]
                          , validation_data=val_gen
                          , validation_freq=1)
        
    def evaluate(self, mode=MODE_VAL, result_saving=False):
        """Evaluate.
        
        Parameters
        ----------
        mode: Integer.
            Data mode (default: MODE_VAL).
        result_saving: Boolean.
            Result saving flag (default: False).
        Returns
        -------
        Mean iou.
            Scalar float.
        
        """
        assert hasattr(self, 'model')

        # Initialize the results directory.
        if result_saving:
            if not os.path.isdir(os.path.join(self.resource_path, 'results')):
                os.mkdir(os.path.join(self.resource_path, 'results'))
            else:
                shutil.rmtree(os.path.join(self.resource_path, 'results'))
                os.mkdir(os.path.join(self.resource_path, 'results'))

        if self.conf['resource_type'] == RESOURCE_TYPE_PASCAL_VOC_2012: 
            val_gen = self.TrainingSequencePascalVOC2012(self.conf
                                                        , mode=MODE_VAL)
        elif self.conf['resource_type'] == RESOURCE_TYPE_PASCAL_VOC_2012_EXT:
            valGen = self.TrainingSequencePascalVOC2012Ext(self.conf
                                                    , mode=MODE_VAL)
        elif self.conf['resource_type'] == RESOURCE_TYPE_GOOGLE_OPEN_IMAGES_V5:
            val_gen = self.TrainingSequenceGoogleOpenImagesV5(self.conf
                                                        , mode=MODE_VAL)
        else:
            raise ValueError('resource type is not valid.')
        
        assert 'tr_step' in self.hps.keys() or 'val_step' in self.hps.keys()
        step = self.hps['val_step'] if mode == MODE_VAL else self.hps['tr_step']
        
        use_multiprocessing = False
        max_queue_size = self.conf['max_queue_size']
        workers = self.conf['workers']
        shuffle = False
                   
        # Check exception.
        if not isinstance(valGen, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multi processing, use the instance of Sequence.'))
        
        enq=None
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
            
            c_miou = MeanIoUExt(num_classes=self.nn_arch['num_classes'])
            pbar = tqdm(range(step))                                    
            for s_i in pbar: #?
                images, labels = next(output_generator)
                results = self.model.predict(images)
                                   
                c_miou.update_state(labels, results)
                pbar.set_description("Mean IOU: {}".format(c_miou.result().numpy()))
                
                # Save result images.
                if result_saving:
                    results = np.argmax(results, axis=-1) * 255. / self.nn_arch['num_classes']
                    results = np.tile(np.expand_dims(results, axis=-1), (1, 1, 1, 3))
                    labels = np.argmax(labels, axis=-1) * 255. / self.nn_arch['num_classes']
                    labels = np.tile(np.expand_dims(labels, axis=-1), (1, 1, 1, 3)) 
                    
                    for b_i in range(self.hps['batch_size']):
                        image = (images[b_i] + 1.0) * 0.5 * 255.
                        image = image.astype('uint8')
                        label = labels[b_i].astype('uint8')
                        result = results[b_i].astype('uint8')
                        overlay_result = cv.addWeighted(image, 0.5, result, 0.5, 0.)
                        final_result = np.concatenate([image, label, result, overlay_result], axis=1)
                        imsave(os.path.join(self.resource_path
                                , 'results'
                                , 'result_{0:d}.png'.format(s_i * self.hps['batch_size'] + b_i))
                                , final_result)
        finally:
            try:
                if enq:
                    enq.stop()
            finally:
                pass
        
        print('Mean iou: {}'.format(c_miou))
        return c_miou

    def test(self):
        """Test."""

        # Initialize the results directory
        if not os.path.isdir(os.path.join(self.resource_path, 'test_results')):
            os.mkdir(os.path.join(self.resource_path, 'test_results'))
        else:
            shutil.rmtree(os.path.join(self.resource_path, 'test_results'))
            os.mkdir(os.path.join(self.resource_path, 'test_results'))

        if self.conf['resource_type'] == RESOURCE_TYPE_PASCAL_VOC_2012: 
            testGen = self.TrainingSequencePascalVOC2012(self.conf
                                                    , mode=MODE_TEST)
        elif self.conf['resource_type'] == RESOURCE_TYPE_PASCAL_VOC_2012_EXT:
            testGen = self.TrainingSequencePascalVOC2012Ext(self.conf
                                                    , mode=MODE_TEST)
        elif self.conf['resource_type'] == RESOURCE_TYPE_GOOGLE_OPEN_IMAGES_V5:
            testGen = self.TrainingSequenceGoogleOpenImagesV5(self.conf
                                                        , mode=MODE_TEST)
        else:
            raise ValueError('resource type is not valid.')

        step = self.hps['test_step']
        
        use_multiprocessing = False
        max_queue_size = 80
        workers = 4
        shuffle = False
                   
        # Check exception.
        if not isinstance(testGen, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multi processing, use the instance of Sequence.'))
        
        try:        
            # Get the output generator.
            if workers > 0:
                if isinstance(testGen, Sequence):
                    enq = OrderedEnqueuer(testGen
                                      , use_multiprocessing=use_multiprocessing
                                      , shuffle=shuffle)
                else:
                    enq = GeneratorEnqueuer(testGen
                                            , use_multiprocessing=use_multiprocessing)
                    
                enq.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enq.get()
            else:
                if isinstance(testGen, Sequence):
                    output_generator = iter_sequence_infinite(testGen)
                else:
                    output_generator = testGen
            
            pbar = tqdm(range(step))                                    
            for s_i in pbar: #?
                images, file_names = next(output_generator)
                results = self.model.predict(images)
                                   
                # Save result images.
                results = np.argmax(results, axis=-1)
                
                for i in range(self.hps['batch_size']):
                    imsave(os.path.join(self.resource_path
                                        , 'test_results'
                                        , file_names[i].split('.')[0] + '.png')
                            , results[i].astype('uint8')) #?
        finally:
            try:
                if enq is not None:
                    enq.stop()
            finally:
                pass        

    def convert_to_tf_lite(self):
        """Convert the model to the tf lite model and save the tf lite model as a binary format."""
        assert hasattr(self, 'model')
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        '''
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS
                                               , tf.lite.OpsSet.SELECT_TF_OPS
                                               , tf.lite.constants.FLOAT16]
        '''
        
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]        
        tflite_model = converter.convert()
        
        with open(os.path.join(self.resource_path, self.TF_LITE_MODEL_PATH), 'wb') as f:
            f.write(tflite_model)
            
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
        else:
            onehots = self.model.predict(images)
                        
        return np.argmax(onehots, axis=-1)

    class TrainingSequenceGoogleOpenImagesV5(Sequence):
        """Training data set sequence for google open images v5."""
                
        def __init__(self, conf, mode=MODE_TRAIN):
            """
            Parameters
            ----------
            conf: Dictionary.
                Configuration.
            mode: String.
                Training or validation mode.
            """
            self.conf = conf
            self.resource_path = self.conf['resource_path']
            self.hps = self.conf['hps']
            self.nn_arch = self.conf['nn_arch']
            self.val_ratio = self.hps['val_ratio']
            self.mode = mode

            if self.mode == MODE_TRAIN:
                df = pd.read_csv(os.path.join(self.resource_path, 'train_valid-annotation-object-segmentation.csv')) #?
                self.image_dir_path = os.path.join(self.resource_path, 'train')
                self.label_dir_path = os.path.join(self.resource_path, 'train-masks')
            elif self.mode == MODE_VAL:
                df = pd.read_csv(os.path.join(self.resource_path, 'validation-annotation-object-segmentation.csv'))
                self.image_dir_path = os.path.join(self.resource_path, 'validation')
                self.label_dir_path = os.path.join(self.resource_path, 'validation-masks')
            elif self.mode == MODE_TEST:
                df = pd.read_csv(os.path.join(self.resource_path, 'test-annotation-object-segmentation.csv'))
                self.image_dir_path = os.path.join(self.resource_path, 'test')
                self.label_dir_path = os.path.join(self.resource_path, 'test-masks')               
            else:
                raise ValueError('The mode must be MODE_TRAIN or MODE_VAL.')
            
            df = df.iloc[:, 1:]
            
            self.class_df = pd.read_csv(os.path.join(self.resource_path, 'class-description-boxable.csv'))
            self.class_df.columns = ['index_class', 'semantic_class']
            
            self.ic2sc = {}
            ic2sc_g = {}
            self.sc2ic = {}
            self.ic2in = {}
            self.sc2in = {}
            
            index_num = 0
            for i in range(self.class_df.shape[0]):
                ic2sc_g[self.class_df.iloc[i, 0]] = self.class_df.iloc[i, 1]
                
                if GOIV5_SPECIFIC_SET.issuperset(self.class_df.iloc[i, 1]): 
                    self.ic2sc[self.class_df.iloc[i, 0]] = self.class_df.iloc[i, 1]
                    self.sc2ic[self.class_df.iloc[i, 1]] = self.class_df.iloc[i, 0]
                    self.ic2in[self.class_df.iloc[i, 0]] = index_num + 1
                    self.sc2in[self.class_df.iloc[i, 1]] = index_num + 1
            
            # Extract specific classes.
            self.df = pd.DataFrame(columns=df.columns)
            
            for i in range(df.shape[0]):
                ic = self.df.iloc[i, 2]
                sc = ic2sc_g[ic]
                
                if GOIV5_SPECIFIC_SET.issuperset(sc):
                    self.df = self.df.append(self.df.iloc[i]) 
             
            self.total_samples = self.df.shape[0]           
            self.batch_size = self.hps['batch_size']
            
            if self.mode == MODE_TRAIN:
                self.hps['tr_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['tr_step'] + 1
                else:
                    self.temp_step = self.hps['tr_step']
            elif self.mode == MODE_VAL:
                self.hps['val_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['val_step'] + 1
                else:
                    self.temp_step = self.hps['val_step']
            elif self.mode == MODE_TEST:
                self.hps['test_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['test_step'] + 1
                else:
                    self.temp_step = self.hps['test_step']
            else:
                raise ValueError('The mode must be MODE_TRAIN or MODE_VAL.')
                            
        def __len__(self):
            return self.temp_step
        
        def __getitem__(self, index):
            images = []
            labels = []
            file_names = []
            
            # Check the last index.
            if index == (self.temp_step - 1):
                for bi in range(index * self.batch_size, self.df.shape[0]):
                    file_name = self.df.iloc[bi, 0].split('_')[0] + '.jpg'
                    label_name = self.df.iloc[bi, 0]
                    index_class = self.df.iloc[bi, 2]
                    
                    if self.mode == MODE_TEST: 
                        file_names.append(file_name) 
                    #if DEBUG: print(file_name )
                    
                    image_path = os.path.join(self.image_dir_path, file_name)
                    
                    # Load image.
                    image = imread(image_path)
                    image = 2.0 * (image / 255 - 0.5) # Normalization to (-1, 1).
                                                             
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    image, w, h, pad_t, pad_l, pad_b, pad_r \
                        = resize_image_to_target_symmeric_size(image
                                                               , self.nn_arch['image_size']
                                                               , device=self.conf['prepro_device'])
                    images.append(image)
                    
                    if self.mode != MODE_TEST:    
                        # Load label.
                        label_path = os.path.join(self.label_dir_path, label_name) #?
                        
                        label = np.expand_dims(imread(label_path), axis=-1)
                        index_num = self.ic2in[index_class]
                        label[label == 1.0] = index_num
                                                                 
                        # Adjust the original label size into the normalized label size according to the ratio of width, height.
                        label, w, h, pad_t, pad_l, pad_b, pad_r \
                            = resize_image_to_target_symmeric_size(label
                                                                   , self.nn_arch['image_size']
                                                                   , device=self.conf['prepro_device'])
                        
                        # Convert label to one hot label.
                        #label = np.expand_dims(label, axis=-1)
                        #if self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        label = get_one_hot(label, self.nn_arch['num_classes'])
                        
                        labels.append(label)          
            else:
                for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                    file_name = self.df.iloc[bi, 0].split('_')[0] + '.jpg'
                    label_name = self.df.iloc[bi, 0]
                    index_class = self.df.iloc[bi, 2]
                    
                    if self.mode == MODE_TEST: 
                        file_names.append(file_name) 
                    #if DEBUG: print(file_name )
                    
                    image_path = os.path.join(self.image_dir_path, file_name)
                    
                    # Load image.
                    image = imread(image_path)
                    image = 2.0 * (image / 255 - 0.5) # Normalization to (-1, 1).
                                                             
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    image, w, h, pad_t, pad_l, pad_b, pad_r \
                        = resize_image_to_target_symmeric_size(image
                                                               , self.nn_arch['image_size']
                                                               , device=self.conf['prepro_device'])
                    images.append(image)

                    if self.mode != MODE_TEST:    
                        # Load label.
                        label_path = os.path.join(self.label_dir_path, label_name) #?
                        
                        label = np.expand_dims(imread(label_path), axis=-1)
                        index_num = self.ic2in[index_class]
                        label[label == 1.0] = index_num
                                                                 
                        # Adjust the original label size into the normalized label size according to the ratio of width, height.
                        label, w, h, pad_t, pad_l, pad_b, pad_r \
                            = resize_image_to_target_symmeric_size(label
                                                                   , self.nn_arch['image_size']
                                                                   , device=self.conf['prepro_device'])
                        
                        # Convert label to one hot label.
                        #label = np.expand_dims(label, axis=-1)
                        #if self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        label = get_one_hot(label, self.nn_arch['num_classes'])
                        
                        labels.append(label) 
                                                                         
            return (np.asarray(images), np.asarray(labels)) \
                    if self.mode != MODE_TEST else (np.asarray(images), file_names)

    class TrainingSequencePascalVOC2012Ext(Sequence):
        """Training data set sequence extension for Pascal VOC 2012."""
                
        def __init__(self, conf, mode=MODE_TRAIN):
            """
            Parameters
            ----------
            conf: Dictionary.
                Configuration.
            mode: String.
                Training or validation mode.
            """
            self.conf = conf
            self.resource_path = self.conf['resource_path']
            self.hps = self.conf['hps']
            self.nn_arch = self.conf['nn_arch']
            self.val_ratio = self.hps['val_ratio']
            self.mode = mode

            if self.mode == MODE_TRAIN or self.mode == MODE_VAL:
                with open(os.path.join(self.resource_path
                                       , 'VOCdevkit'
                                       , 'VOC2012'
                                       , 'ImageSets'
                                       , 'Segmentation'
                                       , 'train_aug_val.txt')) as f:
                    self.file_names = f.readlines()[:] #?
            elif self.mode == MODE_TEST:
                with open(os.path.join(self.resource_path
                                       , 'pascal-voc-2012-test'
                                       , 'VOCdevkit'
                                       , 'VOC2012'
                                       , 'ImageSets'
                                       , 'Segmentation'
                                       , 'test.txt')) as f:
                    self.file_names = f.readlines()[:100] #?
            else:
                raise ValueError('The mode must be MODE_TRAIN or MODE_VAL.')
            
            # Remove \n.
            for i in range(len(self.file_names)):
                self.file_names[i] = self.file_names[i][:-1]

            if self.mode == MODE_TRAIN:
                self.file_names = self.file_names[:int(len(self.file_names) * (1. - self.val_ratio))]    
                self.total_samples = len(self.file_names)
            elif self.mode == MODE_VAL:
                self.file_names = self.file_names[int(len(self.file_names) * (1. - self.val_ratio)):]    
                self.total_samples = len(self.file_names)
            
            if self.mode == MODE_TEST:
                self.image_dir_path = os.path.join(self.resource_path
                                           , 'pascal-voc-2012-test'
                                           , 'VOCdevkit'
                                           , 'VOC2012'
                                           , 'JPEGImages')
            else:
                self.image_dir_path = os.path.join(self.resource_path
                                           , 'VOCdevkit'
                                           , 'VOC2012'
                                           , 'JPEGImages')
                self.label_dir_path = os.path.join(self.resource_path
                                           , 'VOCdevkit'
                                           , 'VOC2012'
                                           , 'SegmentationClassAug')
            
            self.batch_size = self.hps['batch_size']
            
            if self.mode == MODE_TRAIN:
                self.hps['tr_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['tr_step'] + 1
                else:
                    self.temp_step = self.hps['tr_step']
            elif self.mode == MODE_VAL:
                self.hps['val_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['val_step'] + 1
                else:
                    self.temp_step = self.hps['val_step']
            elif self.mode == MODE_TEST:
                self.hps['test_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['test_step'] + 1
                else:
                    self.temp_step = self.hps['test_step']
            else:
                raise ValueError('The mode must be MODE_TRAIN or MODE_VAL.')
                            
        def __len__(self):
            return self.temp_step
        
        def __getitem__(self, index):
            images = []
            labels = []
            file_names = []
            
            # Check the last index.
            if index == (self.temp_step - 1):
                for bi in range(index * self.batch_size, len(self.file_names)):
                    file_name = self.file_names[bi]
                    if self.mode == MODE_TEST: 
                        file_names.append(file_name) 
                    #if DEBUG: print(file_name )
                    
                    image_path = os.path.join(self.image_dir_path, file_name + '.jpg')
                    
                    # Load image.
                    image = imread(image_path)
                    image = 2.0 * (image / 255 - 0.5) # Normalization to (-1, 1).
                                                             
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    image, w, h, pad_t, pad_l, pad_b, pad_r \
                        = resize_image_to_target_symmeric_size(image
                                                               , self.nn_arch['image_size']
                                                               , device=self.conf['prepro_device'])
                    images.append(image)
                    
                    if self.mode != MODE_TEST:    
                        # Load label.
                        label_path = os.path.join(self.label_dir_path, file_name + '.png') #?
                        
                        label = np.expand_dims(imread(label_path), axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                                                                 
                        # Adjust the original label size into the normalized label size according to the ratio of width, height.
                        label, w, h, pad_t, pad_l, pad_b, pad_r \
                            = resize_image_to_target_symmeric_size(label
                                                                   , self.nn_arch['image_size']
                                                                   , device=self.conf['prepro_device'])

                        # Convert label to one hot label.
                        #label = np.expand_dims(label, axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                        #if self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        label = get_one_hot(label, self.nn_arch['num_classes'])
                        
                        labels.append(label)          
            else:
                for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                    file_name = self.file_names[bi]
                    if self.mode == MODE_TEST:
                        file_names.append(file_name)
                        # if DEBUG: print(file_name )

                    image_path = os.path.join(self.image_dir_path, file_name + '.jpg')

                    # Load image.
                    image = imread(image_path)
                    image = 2.0 * (image / 255 - 0.5)  # Normalization to (-1, 1).

                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    image, w, h, pad_t, pad_l, pad_b, pad_r \
                        = resize_image_to_target_symmeric_size(image
                                                               , self.nn_arch['image_size']
                                                               , device=self.conf['prepro_device'])
                    images.append(image)

                    if self.mode != MODE_TEST:
                        # Load label.
                        label_path = os.path.join(self.label_dir_path, file_name + '.png')  # ?

                        label = np.expand_dims(imread(label_path), axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0

                        # Adjust the original label size into the normalized label size according to the ratio of width, height.
                        label, w, h, pad_t, pad_l, pad_b, pad_r \
                            = resize_image_to_target_symmeric_size(label
                                                                   , self.nn_arch['image_size']
                                                                   , device=self.conf['prepro_device'])

                        # Convert label to one hot label.
                        # label = np.expand_dims(label, axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                        # if self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        label = get_one_hot(label, self.nn_arch['num_classes'])

                        labels.append(label)

            return (np.asarray(images), np.asarray(labels)) \
                    if self.mode != MODE_TEST else (np.asarray(images), file_names)

    class TrainingSequencePascalVOC2012(Sequence):
        """Training data set sequence for Pascal VOC 2012."""
                
        def __init__(self, conf, mode=MODE_TRAIN):
            """
            Parameters
            ----------
            conf: Dictionary.
                Configuration.
            mode: String.
                Training or validation mode.
            """
            self.conf = conf
            self.resource_path = self.conf['resource_path']
            self.hps = self.conf['hps']
            self.nn_arch = self.conf['nn_arch']
            self.val_ratio = self.hps['val_ratio']
            self.mode = mode
            
            if self.mode == MODE_TRAIN:
                with open(os.path.join(self.resource_path
                                       , 'VOCdevkit'
                                       , 'VOC2012'
                                       , 'ImageSets'
                                       , 'Segmentation'
                                       , 'train_aug.txt')) as f:
                    self.file_names = f.readlines() #?
            elif self.mode == MODE_VAL:
                with open(os.path.join(self.resource_path
                                       , 'VOCdevkit'
                                       , 'VOC2012'
                                       , 'ImageSets'
                                       , 'Segmentation'
                                       , 'val.txt')) as f:
                    self.file_names = f.readlines() #? 
            elif self.mode == MODE_TEST:
                with open(os.path.join(self.resource_path
                                       , 'pascal-voc-2012-test'
                                       , 'VOCdevkit'
                                       , 'VOC2012'
                                       , 'ImageSets'
                                       , 'Segmentation'
                                       , 'test.txt')) as f:
                    self.file_names = f.readlines() #?                
            else:
                raise ValueError('The mode must be MODE_TRAIN or MODE_VAL.')
            
            # Remove \n.
            for i in range(len(self.file_names)):
                self.file_names[i] = self.file_names[i][:-1]
                
            self.total_samples = len(self.file_names)
            
            if self.mode == MODE_TEST:
                self.image_dir_path = os.path.join(self.resource_path
                                           , 'pascal-voc-2012-test'
                                           , 'VOCdevkit'
                                           , 'VOC2012'
                                           , 'JPEGImages')
            else:
                self.image_dir_path = os.path.join(self.resource_path
                                           , 'VOCdevkit'
                                           , 'VOC2012'
                                           , 'JPEGImages')
                self.label_dir_path = os.path.join(self.resource_path
                                           , 'VOCdevkit'
                                           , 'VOC2012'
                                           , 'SegmentationClassAug')
            
            self.batch_size = self.hps['batch_size']
            
            if self.mode == MODE_TRAIN:
                self.hps['tr_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['tr_step'] + 1
                else:
                    self.temp_step = self.hps['tr_step']
            elif self.mode == MODE_VAL:
                self.hps['val_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['val_step'] + 1
                else:
                    self.temp_step = self.hps['val_step']
            elif self.mode == MODE_TEST:
                self.hps['test_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['test_step'] + 1
                else:
                    self.temp_step = self.hps['test_step']
            else:
                raise ValueError('The mode must be MODE_TRAIN or MODE_VAL.')
                            
        def __len__(self):
            return self.temp_step
        
        def __getitem__(self, index):
            images = []
            labels = []
            file_names = []
            
            # Check the last index.
            if index == (self.temp_step - 1):
                for bi in range(index * self.batch_size, len(self.file_names)):
                    file_name = self.file_names[bi]
                    if self.mode == MODE_TEST: 
                        file_names.append(file_name) 
                    #if DEBUG: print(file_name )
                    
                    image_path = os.path.join(self.image_dir_path, file_name + '.jpg')
                    
                    # Load image.
                    image = imread(image_path)
                    image = 2.0 * (image / 255 - 0.5) # Normalization to (-1, 1).
                                                             
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    image, w, h, pad_t, pad_l, pad_b, pad_r \
                        = resize_image_to_target_symmeric_size(image
                                                               , self.nn_arch['image_size']
                                                               , device=self.conf['prepro_device'])
                    images.append(image)
                    
                    if self.mode != MODE_TEST:    
                        # Load label.
                        label_path = os.path.join(self.label_dir_path, file_name + '.png') #?
                        
                        label = np.expand_dims(imread(label_path), axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                                                                 
                        # Adjust the original label size into the normalized label size according to the ratio of width, height.
                        label, w, h, pad_t, pad_l, pad_b, pad_r \
                            = resize_image_to_target_symmeric_size(label
                                                                   , self.nn_arch['image_size']
                                                                   , device=self.conf['prepro_device'])
                        
                        # Convert label to one hot label.
                        #label = np.expand_dims(label, axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                        #if self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        label = get_one_hot(label, self.nn_arch['num_classes'])
                        
                        labels.append(label)          
            else:
                for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                    file_name = self.file_names[bi]
                    if self.mode == MODE_TEST: 
                        file_names.append(file_name) 
                    #if DEBUG: print(file_name )
                    
                    image_path = os.path.join(self.image_dir_path, file_name + '.jpg')
                    
                    # Load image.
                    image = imread(image_path)
                    image = 2.0 * (image / 255 - 0.5) # Normalization to (-1, 1).
                                                             
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    image, w, h, pad_t, pad_l, pad_b, pad_r \
                        = resize_image_to_target_symmeric_size(image
                                                               , self.nn_arch['image_size']
                                                               , device=self.conf['prepro_device'])
                    images.append(image)
                    
                    if self.mode != MODE_TEST:    
                        # Load label.
                        label_path = os.path.join(self.label_dir_path, file_name + '.png') #?
                        
                        label = np.expand_dims(imread(label_path), axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                                                                 
                        # Adjust the original label size into the normalized label size according to the ratio of width, height.
                        label, w, h, pad_t, pad_l, pad_b, pad_r \
                            = resize_image_to_target_symmeric_size(label
                                                                   , self.nn_arch['image_size']
                                                                   , device=self.conf['prepro_device'])
                        
                        # Convert label to one hot label.
                        #label = np.expand_dims(label, axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                        #if self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        label = get_one_hot(label, self.nn_arch['num_classes'])
                        
                        labels.append(label)
                                                                         
            return (np.asarray(images), np.asarray(labels)) \
                    if self.mode != MODE_TEST else (np.asarray(images), file_names)
                          
def main():
    """Main."""

    # Initialize random generators.
    seed = int(time.time())
    seed = 1024
    print(f'Seed:{seed}')
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load configuration.
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
        ss.evaluate(mode=conf['eval_data_mode'], result_saving=conf['eval_result_saving'])
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif conf['mode'] == 'test':
        # Test.
        ss = SemanticSegmentation(conf)
        
        ts = time.time()
        ss.test()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif conf['mode'] == 'convert_to_tf_lite':
        # Convert the model into the tf lite model. 
        ss = SemanticSegmentation(conf)
        
        ts = time.time()
        ss.convert_to_tf_lite()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))        
    else:
        raise ValueError('mode is not valid.')                        
             
if __name__ == '__main__':
    main()
