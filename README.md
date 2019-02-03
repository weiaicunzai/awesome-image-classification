

# Awesome - Image Classification

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of deep learning image classification papers and codes since 2014, Inspired by [awesome-object-detection](https://github.com/amusi/awesome-object-detection), [deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection) and [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers).

## Background

I believe image classification is a great start point before diving into other computer vision fields, espacially
for begginers who know nothing about deep learning. When I started to learn computer vision, I've made a lot of mistakes, I wish someone could have told me that which paper I should start with back then. There doesn't seem to have a repository to have a list of image classification papers like [deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection) until now. Therefore, I decided to make a repository
of a list of deep learning image classification papers and codes to help others. My personal advice for people who
know nothing about deep learning, try to start with vgg, then googlenet, resnet, feel free to continue reading other listed or switch to other fields after you are finished.

I also have a repository of pytorch implementation of all the image classification networks, you can check out [here](git@github.com:weiaicunzai/pytorch-cifar.git).

## Performance Table

For simplicity reason, I only listed the top1 and top5 accuracy on ImageNet. Note that this does not necessarily mean one network
is better than another when the acc is higher, cause some networks are focused on reducing the model complexity instead of accuracy.

- ConvNet: the name of covolutional net
- ImageNet top1 acc: best top1 accuracy on ImageNet 
- ImageNet top5 acc: best top5 accuracy on ImageNet 
- Published In: which conference or journal the paper is published in.

|       ConvNet          | ImageNet top1 acc | ImageNet top5 acc |   Published In     |
|:----------------------:|:-----------------:|:-----------------:|:------------------:|
|         Vgg            |      76.3         |       93.2        |      ICLR2015      |   
|      GoogleNet         |       -           |       93.33       |      CVPR2015      |   
|      PReLU-nets        |       -           |       95.06       |      ICCV2015      |   
|        ResNet          |       -           |       96.43       |      CVPR2015      |   
|     Inceptionv3        |      82.8         |       96.42       |      CVPR2016      |   
|  Inception-ResNet-v2   |      83.5         |       96.92       |      AAAI2016      |   
|         RiR            |       -           |         -         |  ICLR Workshop2016 |   
|Stochastic Depth ResNet |      78.02        |         -         |      ECCV2016      |   
|         WRN            |      78.1         |       94.21       |      BMVC2016      |   
|      DenseNet          |      79.2         |       94.71       |      ECCV2017      |   
|     FractalNet         |      75.8         |       92.61       |      ICLR2017      |   
|       ResNext          |       -           |       96.97       |      CVPR2017      |   
|       ResNext          |       -           |       96.97       |      CVPR2017      |   