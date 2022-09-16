# Ships Classification using PyTorch

This project was carried out for the [Leonardo Labs Kaggle Competition](https://www.kaggle.com/competitions/sapienza-training-camp-2022/overview) and was able to achieve the 2nd place with a private score of 0.98102. The goal was to classify images of ships according to their type (7 different classes). 

## Approach

The strategy chosen was to train different neural network architectures and then obtain a single prediction by majority voting. Other strategies that have been adopted include training an ensemble of neural networks after combining the outputs and the use of **Visual Transformers**. 

## Dataset

The dataset can be downloaded with the following command:

```python
!gdown -q 1hukMWTFj2aSqx2jBh42R-Y6UXrSw60Nj
```

## Setup

### Loss, Optimizer, Scheduler, Epoques and Batch Size

The adopted loss criterion was the [Cross-Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
The Optimizer was [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) for the first 10 epoques to get closer to the local maxima and then [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD) for 5 more epoques to have a smoother convergence with a [Lambda scheduler](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html).
We chose a small batch size equal to 8 in order to reduce the computational time and have a faster convergence.

### The models

| Models       | Paper           | Link  |
| :-------------: |:-------------:| :-----:|
| ViT B 16     | <https://arxiv.org/abs/2010.11929> | [ViT pretrained (https://github.com/lukemelas/PyTorch-Pretrained-ViT)|
| Resnet152      | <https://arxiv.org/abs/1512.03385>      |  [Resnet152 PyTorch] (https://pytorch.org/vision/0.12/generated/torchvision.models.resnet152.html) |
| ConvNeXt | <https://arxiv.org/abs/2201.03545>      |  [ConvNeXt PyTorch] (https://pytorch.org/vision/stable/models/convnext.html) |
| ResNeXt | <https://arxiv.org/abs/1611.05431>      | [ResNeXt PyTorch] (https://pytorch.org/hub/pytorch_vision_resnext/)|
| SE-ResNeXt | <https://arxiv.org/abs/1709.01507v4>      |   [SE-ResNeXt Timm] (https://rwightman.github.io/pytorch-image-models/models/seresnext/) |
| Xception | <https://arxiv.org/abs/1610.02357>      |  [Xception Timm] (https://rwightman.github.io/pytorch-image-models/models/xception/) |




