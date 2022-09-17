**To download the pretrained models use the following commands:**

```python
!gdown 1Fydxv8Jq7dkPguG_wdn3G2ZVOybejjzL # pretrained xception
!gdown 1B5wHW6jlPVNMq-rq65SPiL4AXDDCnKzN # pretrained vit
!gdown 1fz93kOaXPf-Y0WF3udv0qt2PhaqPYMnQ # pretrained resnext
!gdown 11-nhfpP36MkvzB7sGfZhkgro1nvg3Z5Y # pretrained resnet
!gdown 1acCkBDUj0KBbsbZmVqVamAbu1rHagr4u # pretrained convnext
!gdown
```

**Then to load the model in your notebook:**

```python
model = torch.load('path_to_the_model.csv')
```
