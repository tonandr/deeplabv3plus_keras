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
import random

import numpy as np
import cv2 as cv
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dropout
from tensorflow.keras.layers import Concatenate, Lambda, Activation, AveragePooling2D, SeparableConv2D
from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras import optimizers
from tensorflow.keras.applications import MobileNetV2, Xception
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite

from ku.metrics_ext import MeanIoUExt
from ku.loss_ext import CategoricalCrossentropyWithLabelGT

#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Constants.
DEBUG = True
MLFLOW_USAGE = False
NUM_CLASSES = 21

MODE_TRAIN = 0
MODE_VAL = 1
MODE_TEST = 2

BASE_MODEL_MOBILENETV2 = 0
BASE_MODEL_XCEPTION = 1

class SemanticSegmentation(object):
    """Keras Semantic segmentation model of DeeplabV3+"""
    
    # Constants.
    #MODEL_PATH = 'semantic_segmentation_deeplabv3plus'
    MODEL_PATH = 'semantic_segmentation_deeplabv3plus.h5'
    #MODEL_PATH = 'semantic_segmentation_deeplabv3plus_is224_lr0_0001_ep344.h5'

    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: dictionary
            Semantic segmentation model configuration dictionary.
        """
        
        # Check exception.
        assert conf['nn_arch']['output_stride'] == 8 or conf['nn_arch']['output_stride'] == 16 
        
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
            with CustomObjectScope({'CategoricalCrossentropyWithLabelGT':CategoricalCrossentropyWithLabelGT,
                                    'MeanIoUExt': MeanIoUExt}): 
                if self.conf['multi_gpu']:
                    self.model = load_model(self.MODEL_PATH)
                    
                    self.parallel_model = multi_gpu_model(self.model, gpus=self.conf['num_gpus'])
                    self.parallel_model.compile(optimizer=opt
                                                , loss=self.model.losses
                                                , metrics=self.model.metrics)
                else:
                    self.model = load_model(self.MODEL_PATH)
                    #self.model.compile(optimizer=opt, 
                    #           , loss=CategoricalCrossentropyWithLabelGT(num_classes=self.nn_arch['num_classes'])
                    #           , metrics=[MeanIoUExt(num_classes=NUM_CLASSES)]
        else:
            # Design the semantic segmentation model.
            # Load a base model.
            if self.conf['base_model'] == BASE_MODEL_MOBILENETV2:
                # Load mobilenetv2 as the base model.
                mv2 = MobileNetV2(include_top=False) #, depth_multiplier=self.nn_arch['mv2_depth_multiplier'])
                
                if self.nn_arch['output_stride'] == 8:
                    self.base = Model(inputs=mv2.inputs, outputs=mv2.get_layer('block_5_add').output) # Layer satisfying output stride of 8.
                else:
                    self.base = Model(inputs=mv2.inputs, outputs=mv2.get_layer('block_12_add').output) # Layer satisfying output stride of 16.
                
                self.base.trainable = True
                for layer in self.base.layers: layer.trainable = True #?
                
                self.base._init_set_name('base')
            elif self.conf['base_model'] == BASE_MODEL_XCEPTION:
                # Load xception as the base model.
                mv2 = Xception(include_top=False) #, depth_multiplier=self.nn_arch['mv2_depth_multiplier'])
                
                if self.nn_arch['output_stride'] == 8:
                    self.base = Model(inputs=mv2.inputs, outputs=mv2.get_layer('block4_sepconv2_bn').output) # Layer satisfying output stride of 8.
                else:
                    self.base = Model(inputs=mv2.inputs, outputs=mv2.get_layer('block13_sepconv2_bn').output) # Layer satisfying output stride of 16.
                
                self.base.trainable = True
                for layer in self.base.layers: layer.trainable = True #?
                
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
            opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay'])
            
            self.model.compile(optimizer=opt
                               , loss=CategoricalCrossentropyWithLabelGT(num_classes=self.nn_arch['num_classes'])
                               , metrics=[MeanIoUExt(num_classes=NUM_CLASSES)])
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
        #output = Dropout(rate=self.nn_arch['dropout_rate'])(x3)
        output = x3
        
        self.encoder = Model(input_image, output)
        self.encoder._init_set_name('encoder')
        
    def _make_decoder(self):
        """Make decoder."""
        assert hasattr(self, 'base') and hasattr(self, 'encoder')
        
        inputs = self.encoder.outputs
        features = Input(shape=K.int_shape(inputs[0])[1:])
 
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
            output_stride = output_stride / 8 if output_stride == 16 else output_stride / 4
            
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
        """       
        tr_gen = self.TrainingSequencePascalVOC2012(self.raw_data_path
                                                   , self.hps
                                                   , self.nn_arch
                                                   , mode=MODE_TRAIN)
        val_gen = self.TrainingSequencePascalVOC2012(self.raw_data_path
                                                    , self.hps
                                                    , self.nn_arch
                                                    , mode=MODE_VAL)
        """
        tr_gen = self.TrainingSequencePascalVOC2012Ext(self.raw_data_path
                                                   , self.hps
                                                   , self.nn_arch
                                                   , val_ratio=self.hps['val_ratio']
                                                   , mode=MODE_TRAIN)
        val_gen = self.TrainingSequencePascalVOC2012Ext(self.raw_data_path
                                                    , self.hps
                                                    , self.nn_arch
                                                    , val_ratio=self.hps['val_ratio']
                                                    , mode=MODE_VAL)
        
        assert 'tr_step' in self.hps.keys() and 'val_step' in self.hps.keys()
        
        reduce_lr = ReduceLROnPlateau(monitor='loss'
                                      , factor=self.hps['reduce_lr_factor']
                                      , patience=5
                                      , min_lr=1.e-8
                                      , verbose=1)
        model_check_point = ModelCheckpoint(os.path.join(self.raw_data_path, self.MODEL_PATH)
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
        
        if self.conf['multi_gpu']:
            self.parallel_model.fit_generator(tr_gen
                          , steps_per_epoch=self.hps['tr_step'] #?                   
                          , epochs=self.hps['epochs']
                          , verbose=1
                          , max_queue_size=80
                          , workers=4
                          , use_multiprocessing=False
                          , callbacks=[model_check_point, reduce_lr, tensorboard]
                          , validation_data=val_gen
                          , validation_freq=1)
        else:     
            self.model.fit_generator(tr_gen
                          , steps_per_epoch=self.hps['tr_step']                  
                          , epochs=self.hps['epochs']
                          , verbose=1
                          , max_queue_size=80
                          , workers=4
                          , use_multiprocessing=False
                          , callbacks=[model_check_point, reduce_lr, tensorboard]
                          , validation_data=val_gen
                          , validation_freq=1)

        print('Save the model.')
        self.model.save(self.MODEL_PATH, save_format='h5')            
        #self.model.save(os.path.join(self.raw_data_path, self.MODEL_PATH), save_format='tf')
        
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
            if not os.path.isdir(os.path.join(self.raw_data_path, 'results')):
                os.mkdir(os.path.join(self.raw_data_path, 'results'))
            else:
                shutil.rmtree(os.path.join(self.raw_data_path, 'results'))
                os.mkdir(os.path.join(self.raw_data_path, 'results'))

        valGen = self.TrainingSequencePascalVOC2012Ext(self.raw_data_path
                                                    , self.hps
                                                    , self.nn_arch
                                                    , val_ratio=self.hps['val_ratio']
                                                    , mode=mode)
        assert 'tr_step' in self.hps.keys() or 'val_step' in self.hps.keys()
        step = self.hps['val_step'] if mode == MODE_VAL else self.hps['tr_step']
        
        use_multiprocessing = False
        max_queue_size = 80
        workers = 4
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
            
            c_miou = MeanIoUExt(num_classes=NUM_CLASSES)
            pbar = tqdm(range(step))                                    
            for s_i in pbar: #?
                images, labels = next(output_generator)

                if self.conf['multi_gpu']:
                    results = self.parallel_model.predict(images) #?
                else:
                    results = self.model.predict(images)
                                   
                c_miou.update_state(labels, results)
                pbar.set_description("Mean IOU: {}".format(c_miou.result().numpy()))
                
                # Save result images.
                if result_saving:
                    results = np.argmax(results, axis=-1) * 255. / self.nn_arch['num_classes']
                    results = np.tile(np.expand_dims(results, axis=-1), (1, 1, 1, 3))
                    labels = labels * 255. / self.nn_arch['num_classes']
                    labels = np.tile(np.expand_dims(labels, axis=-1), (1, 1, 1, 3)) 
                    
                    for b_i in range(self.hps['batch_size']):
                        image = (images[b_i] + 1.0) * 0.5 * 255.
                        image = image.astype('uint8')
                        label = labels[b_i].astype('uint8')
                        result = results[b_i].astype('uint8')
                        overlay_result = cv.addWeighted(image, 0.5, result, 0.5, 0.)
                        final_result = np.concatenate([image, label, result, overlay_result], axis=1)
                        imsave(os.path.join(self.raw_data_path
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
        if not os.path.isdir(os.path.join(self.raw_data_path, 'test_results')):
            os.mkdir(os.path.join(self.raw_data_path, 'test_results'))
        else:
            shutil.rmtree(os.path.join(self.raw_data_path, 'test_results'))
            os.mkdir(os.path.join(self.raw_data_path, 'test_results'))

        testGen = self.TrainingSequencePascalVOC2012(self.raw_data_path
                                                    , self.hps
                                                    , self.nn_arch
                                                    , mode=MODE_TEST)
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
                images, labels, file_names = next(output_generator)

                if self.conf['multi_gpu']:
                    results = self.parallel_model.predict(images) #?
                else:
                    results = self.model.predict(images)
                                   
                # Save result images.
                results = np.argmax(results, axis=-1)
                
                for i in range(self.hps['batch_size']):
                    imsave(os.path.join(self.raw_data_path
                                        , 'test_results'
                                        , file_names[i].split('.')[0] + '.png')
                            , results[i].astype('uint8')) #?
        finally:
            try:
                if enq is not None:
                    enq.stop()
            finally:
                pass        
            
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

    class TrainingSequencePascalVOC2012Ext(Sequence):
        """Training data set sequence extension for Pascal VOC 2012."""
                
        def __init__(self, raw_data_path, hps, nn_arch, val_ratio=0.1, shuffle=True, mode=MODE_TRAIN):
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
            """
            self.raw_data_path = raw_data_path
            self.hps = hps
            self.nn_arch = nn_arch
            self.val_ratio = val_ratio
            self.shuffle = shuffle
            self.mode = mode
            random.seed(1024)
            
            if self.mode == MODE_TRAIN or self.mode == MODE_VAL:
                with open(os.path.join(self.raw_data_path
                                       , 'VOCdevkit'
                                       , 'VOC2012'
                                       , 'ImageSets'
                                       , 'Segmentation'
                                       , 'train_aug_val.txt')) as f:
                    self.file_names = f.readlines() #?
            elif self.mode == MODE_TEST:
                with open(os.path.join(self.raw_data_path
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
            
            if self.shuffle:
                random.shuffle(self.file_names)
            
            if self.mode == MODE_TRAIN:
                self.file_names = self.file_names[:int(len(self.file_names) * (1. - self.val_ratio))]    
                self.total_samples = len(self.file_names)
            elif self.mode == MODE_VAL:
                self.file_names = self.file_names[int(len(self.file_names) * (1. - self.val_ratio)):]    
                self.total_samples = len(self.file_names)
            
            if self.mode == MODE_TEST:
                self.image_dir_path = os.path.join(self.raw_data_path
                                           , 'pascal-voc-2012-test'
                                           , 'VOCdevkit'
                                           , 'VOC2012'
                                           , 'JPEGImages')
            else:
                self.image_dir_path = os.path.join(self.raw_data_path
                                           , 'VOCdevkit'
                                           , 'VOC2012'
                                           , 'JPEGImages')
                self.label_dir_path = os.path.join(self.raw_data_path
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
        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_NEAREST)
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
                        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_NEAREST)
                        image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                    
                    images.append(image)
                    
                    if self.mode != MODE_TEST:    
                        # Load label.
                        label_path = os.path.join(self.label_dir_path, file_name + '.png') #?
                        
                        label = np.expand_dims(imread(label_path), axis=-1)
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
            
                            label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_NEAREST)
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
                            
                            label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_NEAREST)
                            label = cv.copyMakeBorder(label, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                        
                        # Convert label to one hot label.
                        #label = np.expand_dims(label, axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                        #if self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        
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
        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_NEAREST)
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
                        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_NEAREST)
                        image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                    
                    images.append(image)
                    
                    if self.mode != MODE_TEST:    
                        # Load label.
                        label_path = os.path.join(self.label_dir_path, file_name + '.png') #?
                        
                        label = np.expand_dims(imread(label_path), axis=-1)
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
            
                            label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_NEAREST)
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
                            
                            label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_NEAREST)
                            label = cv.copyMakeBorder(label, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                        
                        # Convert label to one hot label.
                        #label = np.expand_dims(label, axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                        #if self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        
                        labels.append(label)
                                                                         
            return (np.asarray(images), np.asarray(labels)) \
                    if self.mode != MODE_TEST else (np.asarray(images), np.asarray(labels), file_names) 

    class TrainingSequencePascalVOC2012(Sequence):
        """Training data set sequence for Pascal VOC 2012."""
                
        def __init__(self, raw_data_path, hps, nn_arch, mode=MODE_TRAIN):
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
            
            if self.mode == MODE_TRAIN:
                with open(os.path.join(self.raw_data_path
                                       , 'VOCdevkit'
                                       , 'VOC2012'
                                       , 'ImageSets'
                                       , 'Segmentation'
                                       , 'train_aug.txt')) as f:
                    self.file_names = f.readlines() #?
            elif self.mode == MODE_VAL:
                with open(os.path.join(self.raw_data_path
                                       , 'VOCdevkit'
                                       , 'VOC2012'
                                       , 'ImageSets'
                                       , 'Segmentation'
                                       , 'val.txt')) as f:
                    self.file_names = f.readlines() #? 
            elif self.mode == MODE_TEST:
                with open(os.path.join(self.raw_data_path
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
                self.image_dir_path = os.path.join(self.raw_data_path
                                           , 'pascal-voc-2012-test'
                                           , 'VOCdevkit'
                                           , 'VOC2012'
                                           , 'JPEGImages')
            else:
                self.image_dir_path = os.path.join(self.raw_data_path
                                           , 'VOCdevkit'
                                           , 'VOC2012'
                                           , 'JPEGImages')
                self.label_dir_path = os.path.join(self.raw_data_path
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
        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_NEAREST)
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
                        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_NEAREST)
                        image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                    
                    images.append(image)
                    
                    if self.mode != MODE_TEST:    
                        # Load label.
                        label_path = os.path.join(self.label_dir_path, file_name + '.png') #?
                        
                        label = np.expand_dims(imread(label_path), axis=-1)
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
            
                            label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_NEAREST)
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
                            
                            label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_NEAREST)
                            label = cv.copyMakeBorder(label, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                        
                        # Convert label to one hot label.
                        #label = np.expand_dims(label, axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                        #if self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        
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
        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_NEAREST)
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
                        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_NEAREST)
                        image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                    
                    images.append(image)
                    
                    if self.mode != MODE_TEST:    
                        # Load label.
                        label_path = os.path.join(self.label_dir_path, file_name + '.png') #?
                        
                        label = np.expand_dims(imread(label_path), axis=-1)
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
            
                            label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_NEAREST)
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
                            
                            label = cv.resize(label, (w_p, h_p), interpolation=cv.INTER_NEAREST)
                            label = cv.copyMakeBorder(label, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
                        
                        # Convert label to one hot label.
                        #label = np.expand_dims(label, axis=-1)
                        label[label > (self.nn_arch['num_classes'] - 1)] = 0
                        #if self.eval == False : label = get_one_hot(label, self.nn_arch['num_classes'])
                        
                        labels.append(label)
                                                                         
            return (np.asarray(images), np.asarray(labels)) \
                    if self.mode != MODE_TEST else (np.asarray(images), np.asarray(labels), file_names) 
                          
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
        ss.evaluate(mode=conf['eval_data_mode'], result_saving=conf['eval_result_saving'])
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif conf['mode'] == 'test':
        # Evaluate.
        ss = SemanticSegmentation(conf)
        
        ts = time.time()
        ss.test()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))                    
             
if __name__ == '__main__':
    main()
