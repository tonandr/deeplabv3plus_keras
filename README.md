# Simplified Keras deeplabV3+ semantic segmentation model using Xception and MobileNetV2 as base models
## Simplified Keras based deeplabV3+ has been developed via referring to [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) and [the relevant github repository](https://github.com/tensorflow/models/tree/master/research/deeplab).

The deeplabV3+ semantic segmentation model is mainly composed of the encoder and decoder using atrous spatial pooling and separable depthwise convolution. As training data, [the augmented Pascal VOC 2012 data provided by DrSleep](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) is used. These encoder and decoder become much more simplified and modularized, designing ASPP becomes simplified and flexible as the original deeplabv3+ model of deeplab, so you can design ASPP in the json format, and the boundary refinement layer is modularized, so you can use whether using the boundary refinement layer, or not according to your model's performance. 

# Tasks

- [x] The Keras framework is changed into the tensorflow 2.4 Keras framework.
- [x] The class balanced loss is applied.

This project is closed.

# Requirement

The simplified Keras deeplabV3+ semantic segmentation model is developed and tested on Tensorflow 2.4 and Python 3.6. To use it, Tensorflow 2.4 and Python 3.6 must be installed. OS and GPU environments are the Google Colab GPU environment.   

# Installation, Training, Evaluating

[deeplabv3plus_keras jupyter notebook](deeplabv3plus_keras.ipynb)
