

# Awesome - Image Classification

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of deep learning image classification papers and codes since 2014, Inspired by [awesome-object-detection](https://github.com/amusi/awesome-object-detection), [deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection) and [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers).

## Background

I believe image classification is a great start point before diving into other computer vision fields, espacially
for begginers who know nothing about deep learning. When I started to learn computer vision, I've made a lot of mistakes, I wish someone could have told me that which paper I should start with back then. There doesn't seem to have a repository to have a list of image classification papers like [deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection) until now. Therefore, I decided to make a repository
of a list of deep learning image classification papers and codes to help others. My personal advice for people who
know nothing about deep learning, try to start with vgg, then googlenet, resnet, feel free to continue reading other listed papers or switch to other fields after you are finished.

**Note: I also have a repository of pytorch implementation of some of the image classification networks, you can check out [here](https://github.com/weiaicunzai/pytorch-cifar100).**

## Performance Table

For simplicity reason, I only listed the best top1 and top5 accuracy on ImageNet from the papers. Note that this does not necessarily mean one network is better than another when the acc is higher, cause some networks are focused on reducing the model complexity instead of improving accuracy, or some papers only give the single crop results on ImageNet, but others give the model fusion or multicrop results.

- ConvNet: name of the covolution network
- ImageNet top1 acc: best top1 accuracy on ImageNet from the Paper
- ImageNet top5 acc: best top5 accuracy on ImageNet from the Paper
- Published In: which conference or journal the paper was published in.

|         ConvNet            | ImageNet top1 acc | ImageNet top5 acc |   Published In     |
|:--------------------------:|:-----------------:|:-----------------:|:------------------:|
|           Vgg              |      76.3         |       93.2        |      ICLR2015      |
|        GoogleNet           |       -           |       93.33       |      CVPR2015      |
|        PReLU-nets          |       -           |       95.06       |      ICCV2015      |
|          ResNet            |       -           |       96.43       |      CVPR2015      |
|       PreActResNet         |      79.9         |       95.2        |      CVPR2016      |
|       Inceptionv3          |      82.8         |       96.42       |      CVPR2016      |
|       Inceptionv4          |      82.3         |       96.2        |      AAAI2016      |
|    Inception-ResNet-v2     |      82.4         |       96.3        |      AAAI2016      |
|Inceptionv4 + Inception-ResNet-v2|      83.5         |       96.92       |      AAAI2016      |
|           RiR              |       -           |         -         |  ICLR Workshop2016 |
|  Stochastic Depth ResNet   |      78.02        |         -         |      ECCV2016      |
|           WRN              |      78.1         |       94.21       |      BMVC2016      |
|       SqueezeNet           |      60.4         |       82.5        |      arXiv2017([rejected by ICLR2017](https://openreview.net/forum?id=S1xh5sYgx))     |
|          GeNet             |      72.13        |       90.26       |      ICCV2017      |
|         MetaQNN            |       -           |         -         |      ICLR2017      |
|        PyramidNet          |      80.8         |       95.3        |      CVPR2017      |
|         DenseNet           |      79.2         |       94.71       |      ECCV2017      |
|        FractalNet          |      75.8         |       92.61       |      ICLR2017      |
|         ResNext            |       -           |       96.97       |      CVPR2017      |
|         IGCV1              |      73.05        |       91.08       |      ICCV2017      |
| Residual Attention Network |      80.5         |       95.2        |      CVPR2017      |
|        Xception            |       79          |       94.5        |      CVPR2017      |
|        MobileNet           |      70.6         |         -         |      arXiv2017     |
|         PolyNet            |      82.64        |       96.55       |      CVPR2017      |
|           DPN              |       79          |       94.5        |      NIPS2017      |
|        Block-QNN           |      77.4         |       93.54       |      CVPR2018      |
|         CRU-Net            |      79.7         |       94.7        |      IJCAI2018     |
|       ShuffleNet           |      75.3         |         -         |      CVPR2018      |
|       CondenseNet          |      73.8         |       91.7        |      CVPR2018      |
|          NasNet            |      82.7         |       96.2        |      CVPR2018      |
|       MobileNetV2          |      74.7         |         -         |      CVPR2018      |
|         IGCV2              |      70.07        |         -         |      CVPR2018      |
|          hier              |      79.7         |       94.8        |      ICLR2018      |
|         PNasNet            |      82.9         |       96.2        |      ECCV2018      |
|        AmoebaNet           |      83.9         |       96.6        |      arXiv2018     |
|          SENet             |       -           |       97.749      |      CVPR2018      |
|       ShuffleNetV2         |      81.44        |         -         |      ECCV2018      |
|          IGCV3             |      72.2         |         -         |      BMVC2018      |
|         MnasNet            |      76.13        |       92.85       |      CVPR2018      |
|          SKNet             |      80.60        |         -         |      CVPR2019      |
|          DARTS             |      73.3         |       91.3        |      ICLR2019      |
|       ProxylessNAS         |      75.1         |       92.5        |      ICLR2019      |
|       MobileNetV3          |      75.2         |         -         |      arXiv2019     |
|          Res2Net           |      79.2         |       94.37       |      arXiv2019     |
|       EfficientNet         |      84.3         |       97.0        |      ICML2019      |


## Papers&Codes

### VGG
**Very Deep Convolutional Networks for Large-Scale Image Recognition.**
Karen Simonyan, Andrew Zisserman
- pdf: [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
- code: [torchvision : https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)

### GoogleNet
**Going Deeper with Convolutions**
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
- pdf: [https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)
- code: [unofficial-tensorflow : https://github.com/conan7882/GoogLeNet-Inception](https://github.com/conan7882/GoogLeNet-Inception)
- code: [unofficial-caffe : https://github.com/lim0606/caffe-googlenet-bn](https://github.com/lim0606/caffe-googlenet-bn)

### PReLU-nets
**Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification**
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- pdf: [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
- code: [unofficial-chainer : https://github.com/nutszebra/prelu_net](https://github.com/nutszebra/prelu_net)

### ResNet
**Deep Residual Learning for Image Recognition**
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- pdf: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- code: [facebook-torch : https://github.com/facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)
- code: [torchvision : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet.py)
- code: [unofficial-keras : https://github.com/raghakot/keras-resnet](https://github.com/raghakot/keras-resnet)
- code: [unofficial-tensorflow : https://github.com/ry/tensorflow-resnet](https://github.com/ry/tensorflow-resnet)

### PreActResNet
**Identity Mappings in Deep Residual Networks**
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- pdf: [https://arxiv.org/abs/1603.05027](https://arxiv.org/abs/1603.05027)
- code: [facebook-torch : https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua](https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua)
- code: [official : https://github.com/KaimingHe/resnet-1k-layers](https://github.com/KaimingHe/resnet-1k-layers)
- code: [unoffical-pytorch : https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py](https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py)
- code: [unoffical-mxnet : https://github.com/tornadomeet/ResNet](https://github.com/tornadomeet/ResNet)

### Inceptionv3
**Rethinking the Inception Architecture for Computer Vision**
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
- pdf: [https://arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567)
- code: [torchvision : https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py)

### Inceptionv4 && Inception-ResNetv2
**Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning**
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
- pdf: [https://arxiv.org/abs/1602.07261](https://arxiv.org/abs/1602.07261)
- code: [unofficial-keras : https://github.com/kentsommer/keras-inceptionV4](https://github.com/kentsommer/keras-inceptionV4)
- code: [unofficial-keras : https://github.com/titu1994/Inception-v4](https://github.com/titu1994/Inception-v4)
- code: [unofficial-keras : https://github.com/yuyang-huang/keras-inception-resnet-v2](https://github.com/yuyang-huang/keras-inception-resnet-v2)

### RiR
**Resnet in Resnet: Generalizing Residual Architectures**
Sasha Targ, Diogo Almeida, Kevin Lyman
- pdf: [https://arxiv.org/abs/1603.08029](https://arxiv.org/abs/1603.08029)
- code: [unofficial-tensorflow : https://github.com/SunnerLi/RiR-Tensorflow](https://github.com/SunnerLi/RiR-Tensorflow)
- code: [unofficial-chainer : https://github.com/nutszebra/resnet_in_resnet](https://github.com/nutszebra/resnet_in_resnet)

### Stochastic Depth ResNet
**Deep Networks with Stochastic Depth**
Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger
- pdf: [https://arxiv.org/abs/1603.09382](https://arxiv.org/abs/1603.09382)
- code: [unofficial-torch : https://github.com/yueatsprograms/Stochastic_Depth](https://github.com/yueatsprograms/Stochastic_Depth)
- code: [unofficial-chainer : https://github.com/yasunorikudo/chainer-ResDrop](https://github.com/yasunorikudo/chainer-ResDrop)
- code: [unofficial-keras : https://github.com/dblN/stochastic_depth_keras](https://github.com/dblN/stochastic_depth_keras)

### WRN
**Wide Residual Networks**
Sergey Zagoruyko, Nikos Komodakis
- pdf: [https://arxiv.org/abs/1605.07146](https://arxiv.org/abs/1605.07146)
- code: [official : https://github.com/szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)
- code: [unofficial-pytorch : https://github.com/xternalz/WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch)
- code: [unofficial-keras : https://github.com/asmith26/wide_resnets_keras](https://github.com/asmith26/wide_resnets_keras)
- code: [unofficial-pytorch : https://github.com/meliketoy/wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)

### SqueezeNet
**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size**
Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
- pdf: [https://arxiv.org/abs/1602.07360](https://arxiv.org/abs/1602.07360)
- code: [torchvision : https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py)
- code: [unofficial-caffe : https://github.com/DeepScale/SqueezeNet](https://github.com/DeepScale/SqueezeNet)
- code: [unofficial-keras : https://github.com/rcmalli/keras-squeezenet](https://github.com/rcmalli/keras-squeezenet)
- code: [unofficial-caffe : https://github.com/songhan/SqueezeNet-Residual](https://github.com/songhan/SqueezeNet-Residual)

### GeNet
**Genetic CNN**
Lingxi Xie, Alan Yuille
- pdf: [https://arxiv.org/abs/1703.01513](https://arxiv.org/abs/1703.01513)
- code: [unofficial-tensorflow : https://github.com/aqibsaeed/Genetic-CNN](https://github.com/aqibsaeed/Genetic-CNN)

### MetaQNN
**Designing Neural Network Architectures using Reinforcement Learning**
Bowen Baker, Otkrist Gupta, Nikhil Naik, Ramesh Raskar
- pdf: [https://arxiv.org/abs/1611.02167](https://arxiv.org/abs/1611.02167)
- code: [official : https://github.com/bowenbaker/metaqnn](https://github.com/bowenbaker/metaqnn)

### PyramidNet
**Deep Pyramidal Residual Networks**
Dongyoon Han, Jiwhan Kim, Junmo Kim
- pdf: [https://arxiv.org/abs/1610.02915](https://arxiv.org/abs/1610.02915)
- code: [official : https://github.com/jhkim89/PyramidNet](https://github.com/jhkim89/PyramidNet)
- code: [unofficial-pytorch : https://github.com/dyhan0920/PyramidNet-PyTorch](https://github.com/dyhan0920/PyramidNet-PyTorch)

### DenseNet
**Densely Connected Convolutional Networks**
Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
- pdf: [https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
- code: [official : https://github.com/liuzhuang13/DenseNet](https://github.com/liuzhuang13/DenseNet)
- code: [unofficial-keras : https://github.com/titu1994/DenseNet](https://github.com/titu1994/DenseNet)
- code: [unofficial-caffe : https://github.com/shicai/DenseNet-Caffe](https://github.com/shicai/DenseNet-Caffe)
- code: [unofficial-tensorflow : https://github.com/YixuanLi/densenet-tensorflow](https://github.com/YixuanLi/densenet-tensorflow)
- code: [unofficial-pytorch : https://github.com/YixuanLi/densenet-tensorflow](https://github.com/YixuanLi/densenet-tensorflow)
- code: [unofficial-pytorch : https://github.com/bamos/densenet.pytorch](https://github.com/bamos/densenet.pytorch)
- code: [unofficial-keras : https://github.com/flyyufelix/DenseNet-Keras](https://github.com/flyyufelix/DenseNet-Keras)

### FractalNet
**FractalNet: Ultra-Deep Neural Networks without Residuals**
Gustav Larsson, Michael Maire, Gregory Shakhnarovich
- pdf: [https://arxiv.org/abs/1605.07648](https://arxiv.org/abs/1605.07648)
- code: [unofficial-caffe : https://github.com/gustavla/fractalnet](https://github.com/gustavla/fractalnet)
- code: [unofficial-keras : https://github.com/snf/keras-fractalnet](https://github.com/snf/keras-fractalnet)
- code: [unofficial-tensorflow : https://github.com/tensorpro/FractalNet](https://github.com/tensorpro/FractalNet)

### ResNext
**Aggregated Residual Transformations for Deep Neural Networks**
Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
- pdf: [https://arxiv.org/abs/1611.05431](https://arxiv.org/abs/1611.05431)
- code: [official : https://github.com/facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnext.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnext.py)
- code: [unofficial-pytorch : https://github.com/prlz77/ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch)
- code: [unofficial-keras : https://github.com/titu1994/Keras-ResNeXt](https://github.com/titu1994/Keras-ResNeXt)
- code: [unofficial-tensorflow : https://github.com/taki0112/ResNeXt-Tensorflow](https://github.com/taki0112/ResNeXt-Tensorflow)
- code: [unofficial-tensorflow : https://github.com/wenxinxu/ResNeXt-in-tensorflow](https://github.com/wenxinxu/ResNeXt-in-tensorflow)

### IGCV1
**Interleaved Group Convolutions for Deep Neural Networks**
Ting Zhang, Guo-Jun Qi, Bin Xiao, Jingdong Wang
- pdf: [https://arxiv.org/abs/1707.02725](https://arxiv.org/abs/1707.02725)
- code [official : https://github.com/hellozting/InterleavedGroupConvolutions](https://github.com/hellozting/InterleavedGroupConvolutions)

### Residual Attention Network
**Residual Attention Network for Image Classification**
Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang
- pdf: [https://arxiv.org/abs/1704.06904](https://arxiv.org/abs/1704.06904)
- code: [official : https://github.com/fwang91/residual-attention-network](https://github.com/fwang91/residual-attention-network)
- code: [unofficial-pytorch : https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch](https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch)
- code: [unofficial-gluon : https://github.com/PistonY/ResidualAttentionNetwork](https://github.com/PistonY/ResidualAttentionNetwork)
- code: [unofficial-keras : https://github.com/koichiro11/residual-attention-network](https://github.com/koichiro11/residual-attention-network)

### Xception
**Xception: Deep Learning with Depthwise Separable Convolutions**
François Chollet
- pdf: [https://arxiv.org/abs/1610.02357](https://arxiv.org/abs/1610.02357)
- code: [unofficial-pytorch : https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py](https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py)
- code: [unofficial-tensorflow : https://github.com/kwotsin/TensorFlow-Xception](https://github.com/kwotsin/TensorFlow-Xception)
- code: [unofficial-caffe : https://github.com/yihui-he/Xception-caffe](https://github.com/yihui-he/Xception-caffe)
- code: [unofficial-pytorch : https://github.com/tstandley/Xception-PyTorch](https://github.com/tstandley/Xception-PyTorch)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py)

### MobileNet
**MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
- pdf: [https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)
- code: [unofficial-tensorflow : https://github.com/Zehaos/MobileNet](https://github.com/Zehaos/MobileNet)
- code: [unofficial-caffe : https://github.com/shicai/MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe)
- code: [unofficial-pytorch : https://github.com/marvis/pytorch-mobilenet](https://github.com/marvis/pytorch-mobilenet)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py)

### PolyNet
**PolyNet: A Pursuit of Structural Diversity in Very Deep Networks**
Xingcheng Zhang, Zhizhong Li, Chen Change Loy, Dahua Lin
- pdf: [https://arxiv.org/abs/1611.05725](https://arxiv.org/abs/1611.05725)
- code: [official : https://github.com/open-mmlab/polynet](https://github.com/open-mmlab/polynet)

### DPN
**Dual Path Networks**
Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng
- pdf: [https://arxiv.org/abs/1707.01629](https://arxiv.org/abs/1707.01629)
- code: [official : https://github.com/cypw/DPNs](https://github.com/cypw/DPNs)
- code: [unoffical-keras : https://github.com/titu1994/Keras-DualPathNetworks](https://github.com/titu1994/Keras-DualPathNetworks)
- code: [unofficial-pytorch : https://github.com/oyam/pytorch-DPNs](https://github.com/oyam/pytorch-DPNs)
- code: [unofficial-pytorch : https://github.com/rwightman/pytorch-dpn-pretrained](https://github.com/rwightman/pytorch-dpn-pretrained)

### Block-QNN
**Practical Block-wise Neural Network Architecture Generation**
Zhao Zhong, Junjie Yan, Wei Wu, Jing Shao, Cheng-Lin Liu
- pdf: [https://arxiv.org/abs/1708.05552](https://arxiv.org/abs/1708.05552)

### CRU-Net
**Sharing Residual Units Through Collective Tensor Factorization in Deep Neural Networks**
Chen Yunpeng, Jin Xiaojie, Kang Bingyi, Feng Jiashi, Yan Shuicheng
- pdf: [https://arxiv.org/abs/1703.02180](https://arxiv.org/abs/1703.02180)
- code [official : https://github.com/cypw/CRU-Net](https://github.com/cypw/CRU-Net)
- code [unofficial-mxnet : https://github.com/bruinxiong/Modified-CRUNet-and-Residual-Attention-Network.mxnet](https://github.com/bruinxiong/Modified-CRUNet-and-Residual-Attention-Network.mxnet)

### ShuffleNet
**ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices**
Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
- pdf: [https://arxiv.org/abs/1707.01083](https://arxiv.org/abs/1707.01083)
- code: [unofficial-tensorflow : https://github.com/MG2033/ShuffleNet](https://github.com/MG2033/ShuffleNet)
- code: [unofficial-pytorch : https://github.com/jaxony/ShuffleNet](https://github.com/jaxony/ShuffleNet)
- code: [unofficial-caffe : https://github.com/farmingyard/ShuffleNet](https://github.com/farmingyard/ShuffleNet)
- code: [unofficial-keras : https://github.com/scheckmedia/keras-shufflenet](https://github.com/scheckmedia/keras-shufflenet)

### CondenseNet
**CondenseNet: An Efficient DenseNet using Learned Group Convolutions**
Gao Huang, Shichen Liu, Laurens van der Maaten, Kilian Q. Weinberger
- pdf: [https://arxiv.org/abs/1711.09224](https://arxiv.org/abs/1711.09224)
- code: [official : https://github.com/ShichenLiu/CondenseNet](https://github.com/ShichenLiu/CondenseNet)
- code: [unofficial-tensorflow : https://github.com/markdtw/condensenet-tensorflow](https://github.com/markdtw/condensenet-tensorflow)

### NasNet
**Learning Transferable Architectures for Scalable Image Recognition**
Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le
- pdf: [https://arxiv.org/abs/1707.07012](https://arxiv.org/abs/1707.07012)
- code: [unofficial-keras : https://github.com/titu1994/Keras-NASNet](https://github.com/titu1994/Keras-NASNet)
- code: [keras-applications : https://github.com/keras-team/keras-applications/blob/master/keras_applications/nasnet.py](https://github.com/keras-team/keras-applications/blob/master/keras_applications/nasnet.py)
- code: [unofficial-pytorch : https://github.com/wandering007/nasnet-pytorch](https://github.com/wandering007/nasnet-pytorch)
- code: [unofficial-tensorflow : https://github.com/yeephycho/nasnet-tensorflow](https://github.com/yeephycho/nasnet-tensorflow)

### MobileNetV2
**MobileNetV2: Inverted Residuals and Linear Bottlenecks**
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
- pdf: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- code: [unofficial-keras : https://github.com/xiaochus/MobileNetV2](https://github.com/xiaochus/MobileNetV2)
- code: [unofficial-pytorch : https://github.com/Randl/MobileNetV2-pytorch](https://github.com/Randl/MobileNetV2-pytorch)
- code: [unofficial-tensorflow : https://github.com/neuleaf/MobileNetV2](https://github.com/neuleaf/MobileNetV2)

### IGCV2
**IGCV2: Interleaved Structured Sparse Convolutional Neural Networks**
Guotian Xie, Jingdong Wang, Ting Zhang, Jianhuang Lai, Richang Hong, Guo-Jun Qi
- pdf: [https://arxiv.org/abs/1804.06202](https://arxiv.org/abs/1804.06202)

### hier
**Hierarchical Representations for Efficient Architecture Search**
Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, Koray Kavukcuoglu
- pdf: [https://arxiv.org/abs/1711.00436](https://arxiv.org/abs/1711.00436)

### PNasNet
**Progressive Neural Architecture Search**
Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, Kevin Murphy
- pdf: [https://arxiv.org/abs/1712.00559](https://arxiv.org/abs/1712.00559)
- code: [tensorflow-slim : https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/pnasnet.py](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/pnasnet.py)
- code: [unofficial-pytorch : https://github.com/chenxi116/PNASNet.pytorch](https://github.com/chenxi116/PNASNet.pytorch)
- code: [unofficial-tensorflow : https://github.com/chenxi116/PNASNet.TF](https://github.com/chenxi116/PNASNet.TF)

### AmoebaNet
**Regularized Evolution for Image Classifier Architecture Search**
Esteban Real, Alok Aggarwal, Yanping Huang, Quoc V Le
- pdf: [https://arxiv.org/abs/1802.01548](https://arxiv.org/abs/1802.01548)
- code: [tensorflow-tpu : https://github.com/tensorflow/tpu/tree/master/models/official/amoeba_net](https://github.com/tensorflow/tpu/tree/master/models/official/amoeba_net)

### SENet
**Squeeze-and-Excitation Networks**
Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
- pdf: [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)
- code: [official : https://github.com/hujie-frank/SENet](https://github.com/hujie-frank/SENet)
- code: [unofficial-pytorch : https://github.com/moskomule/senet.pytorch](https://github.com/moskomule/senet.pytorch)
- code: [unofficial-tensorflow : https://github.com/taki0112/SENet-Tensorflow](https://github.com/taki0112/SENet-Tensorflow)
- code: [unofficial-caffe : https://github.com/shicai/SENet-Caffe](https://github.com/shicai/SENet-Caffe)
- code: [unofficial-mxnet : https://github.com/bruinxiong/SENet.mxnet](https://github.com/bruinxiong/SENet.mxnet)

### ShuffleNetV2
**ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design**
Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
- pdf: [https://arxiv.org/abs/1807.11164](https://arxiv.org/abs/1807.11164)
- code: [unofficial-pytorch : https://github.com/Randl/ShuffleNetV2-pytorch](https://github.com/Randl/ShuffleNetV2-pytorch)
- code: [unofficial-keras : https://github.com/opconty/keras-shufflenetV2](https://github.com/opconty/keras-shufflenetV2)
- code: [unofficial-pytorch : https://github.com/Bugdragon/ShuffleNet_v2_PyTorch](https://github.com/Bugdragon/ShuffleNet_v2_PyTorch)
- code: [unofficial-caff2: https://github.com/wolegechu/ShuffleNetV2.Caffe2](https://github.com/wolegechu/ShuffleNetV2.Caffe2)

### IGCV3
**IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks**
Ke Sun, Mingjie Li, Dong Liu, Jingdong Wang
- pdf: [https://arxiv.org/abs/1806.00178](https://arxiv.org/abs/1806.00178)
- code: [official : https://github.com/homles11/IGCV3](https://github.com/homles11/IGCV3)
- code: [unofficial-pytorch : https://github.com/xxradon/IGCV3-pytorch](https://github.com/xxradon/IGCV3-pytorch)
- code: [unofficial-tensorflow : https://github.com/ZHANG-SHI-CHANG/IGCV3](https://github.com/ZHANG-SHI-CHANG/IGCV3)

### MNasNet
**MnasNet: Platform-Aware Neural Architecture Search for Mobile**
Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Quoc V. Le
- pdf: [https://arxiv.org/abs/1807.11626](https://arxiv.org/abs/1807.11626)
- code: [unofficial-pytorch : https://github.com/AnjieZheng/MnasNet-PyTorch](https://github.com/AnjieZheng/MnasNet-PyTorch)
- code: [unofficial-caffe : https://github.com/LiJianfei06/MnasNet-caffe](https://github.com/LiJianfei06/MnasNet-caffe)
- code: [unofficial-MxNet : https://github.com/chinakook/Mnasnet.MXNet](https://github.com/chinakook/Mnasnet.MXNet)
- code: [unofficial-keras : https://github.com/Shathe/MNasNet-Keras-Tensorflow](https://github.com/Shathe/MNasNet-Keras-Tensorflow)

### SKNet
**Selective Kernel Networks**
Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang
- pdf: [https://arxiv.org/abs/1903.06586](https://arxiv.org/abs/1903.06586)
- code: [official : https://github.com/implus/SKNet](https://github.com/implus/SKNet)

### DARTS
**DARTS: Differentiable Architecture Search**
Hanxiao Liu, Karen Simonyan, Yiming Yang
- pdf: [https://arxiv.org/abs/1806.09055](https://arxiv.org/abs/1806.09055)
- code: [official : https://github.com/quark0/darts](https://github.com/quark0/darts)
- code: [unofficial-pytorch : https://github.com/khanrc/pt.darts](https://github.com/khanrc/pt.darts)
- code: [unofficial-tensorflow : https://github.com/NeroLoh/darts-tensorflow](https://github.com/NeroLoh/darts-tensorflow)

### ProxylessNAS
**ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware**
Han Cai, Ligeng Zhu, Song Han
- pdf: [https://arxiv.org/abs/1812.00332](https://arxiv.org/abs/1812.00332)
- code: [official : https://github.com/mit-han-lab/ProxylessNAS](https://github.com/mit-han-lab/ProxylessNAS)

### MobileNetV3
**Searching for MobileNetV3**
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
- pdf: [https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)
- code: [unofficial-pytorch : https://github.com/xiaolai-sqlai/mobilenetv3](https://github.com/xiaolai-sqlai/mobilenetv3)
- code: [unofficial-pytorch : https://github.com/kuan-wang/pytorch-mobilenet-v3](https://github.com/kuan-wang/pytorch-mobilenet-v3)
- code: [unofficial-pytorch : https://github.com/leaderj1001/MobileNetV3-Pytorch](https://github.com/leaderj1001/MobileNetV3-Pytorch)
- code: [unofficial-pytorch : https://github.com/d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)
- code: [unofficial-caffe : https://github.com/jixing0415/caffe-mobilenet-v3](https://github.com/jixing0415/caffe-mobilenet-v3)
- code: [unofficial-keras : https://github.com/xiaochus/MobileNetV3](https://github.com/xiaochus/MobileNetV3)

### Res2Net
**Res2Net: A New Multi-scale Backbone Architecture**
Shang-Hua Gao, Ming-Ming Cheng, Kai Zhao, Xin-Yu Zhang, Ming-Hsuan Yang, Philip Torr
- pdf: [https://arxiv.org/abs/1904.01169](https://arxiv.org/abs/1904.01169)
- code: [unofficial-pytorch : https://github.com/4uiiurz1/pytorch-res2net](https://github.com/4uiiurz1/pytorch-res2net)
- code: [unofficial-keras : https://github.com/fupiao1998/res2net-keras](https://github.com/fupiao1998/res2net-keras)

### EfficientNet

**EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**
Mingxing Tan, Quoc V. Le
- pdf: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
- code: [unofficial-pytorch : https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- code: [official-tensorflow : https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

