# Keras deeplabV3+ semantic segmentation model using MobileNetV2 as a base model.
## Keras based deeplabV3+ has been developed via referring to [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) and [the relevant github repository](https://github.com/tensorflow/models/tree/master/research/deeplab).

The deeplabV3+ semantic segmentation model is mainly composed of the encoder and decoder using atrous spatial pooling and separable depthwise convolution. 
Focusing on Keras deeplabV3+ with MobileNetV2 as a base model, this encoder and decoder become much more simplified. In the decoder, when using MobileNetV2, 
simple upsampling is applied according to the github repository's source code.

# Tasks
- [x] Encoder develop.
- [x] Decoder develop.
- [x] Training and evaluating with Pasal VOC 2012 dataset.
- [x] Documentation.
- [x] The Keras framework is changed into the tensorflow 2.0 Keras framework.
- [ ] Test and optimize the model.
- [ ] Second documentation.

# Next plan
After the Keras framework is changed into tensorflow 2.0 Keras framework, the model is being tested and optimized in the google Colab environment.
