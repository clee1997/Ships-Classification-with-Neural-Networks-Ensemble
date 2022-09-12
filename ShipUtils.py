import Helper
import torch
from torchvision import datasets, transforms, models, utils
from torch.utils.data import DataLoader
import os
import torch
import torch.optim as optim
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
import tqdm.notebook as tq
import timm
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from sklearn.metrics import f1_score
from glob import glob
from torchvision.datasets import ImageFolder
from PIL import Image
from pytorch_pretrained_vit import ViT
from sklearn.metrics import confusion_matrix

num_classes = 7
batch_size = 8
num_workers = 2

# WE want exploit the GPU!
device = "cuda"
epochs = 10 # the number of epochs

class ShipDataset:
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        cat = self.df.category_name[index]
        cat_id = self.df.category_id[index]
        file_name = self.df.file_name[index]
        img_path = glob(f'../input/sapienza-training-camp-2022/train/train/*/{self.df.file_name[index]}')[0] # bypass folders
        
  
        with open(img_path, "rb") as fp:
            img = Image.open(img_path).convert("RGB")
          
        # transform it to tensor (or eventually augment it)
        img = self.transform(img)
        return img, cat_id

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


def load_df(class_name = 'coast-guard', path = './train_cleaned.csv', test_size = 0.35, closed = False):

    df = pd.read_csv(path, index_col = 0)

    if closed == False:
      df.loc[df['category_name'] != class_name, 'category_id'] = 0
      df.loc[df['category_name'] == class_name, 'category_id'] = 1

    df_train, df_val = train_test_split(df, stratify=df.category_id, test_size=0.35, shuffle = True)
    df_train.reset_index(inplace=True, drop = True)
    df_val.reset_index(inplace=True, drop = True)

    return df_train, df_val


def get_augmented_loaders(df_train, df_val, input_size = (384, 384), rounds = 2):

    # Perform a transformation of a PIL Image
    train_transform = T.Compose([T.Resize(input_size), 
                                 T.ToTensor(), 
                                 T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    t1 = T.Compose([T.Resize(input_size), 
                    T.RandomHorizontalFlip(p = 1),
                    T.ToTensor(), 
                    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    
    t2 = T.Compose([T.Resize(input_size),
                    T.RandomRotation(35),
                    T.ToTensor(), 
                    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    
    t3 = T.Compose([T.Resize(input_size), 
                    T.GaussianBlur(kernel_size=3, sigma=3),
                    T.ToTensor(), 
                    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    
    t4 = T.Compose([T.Resize(input_size), 
                    T.RandomAdjustSharpness(sharpness_factor=2, p = 1),
                    T.RandomPerspective(distortion_scale = 0.3, p = 1),
                    T.ToTensor(), 
                    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    

    train_dataset = ShipDataset(df_train[df_train.category_id == 0].reset_index(drop = True), transform=train_transform)
    val_dataset = ShipDataset(df_val, transform=train_transform)

    aug = ShipDataset(df_train[df_train.category_id != 0].reset_index(drop = True), transform = train_transform)
    aug_1 = ShipDataset(df_train[df_train.category_id != 0].reset_index(drop = True), transform = t1) 
    aug_2 = ShipDataset(df_train[df_train.category_id != 0].reset_index(drop = True), transform = t2)
    aug_3 = ShipDataset(df_train[df_train.category_id != 0].reset_index(drop = True), transform = t3)
    aug_4 = ShipDataset(df_train[df_train.category_id != 0].reset_index(drop = True), transform = t4)

    if rounds == 2: train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_1, aug_2])
    if rounds == 2: train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_1, aug_2])
    if rounds == 3: train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_1, aug_2, aug_3]) 
    if rounds == 4: train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_1, aug_2, aug_3, aug_4]) 

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    print('train samples:', len(train_dataset))
    print('val samples:', len(val_dataset))

    return train_loader, val_loader


def train(train_loader, val_loader, path):

  model = ViT('B_16_imagenet1k', pretrained=True)
  model.fc.out_features = 2
  model.to(device)
  
  cost = torch.nn.CrossEntropyLoss()

  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

  lambda1 = lambda epoch: 0.65 ** epoch
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

  model, train_loss, test_loss = Helper.train(model, train_loader, val_loader, epochs, optimizer, cost, scheduler)

  torch.save(model, path)

  return model


def eval(model, loader):
    y_pred = []
    y_true = []
    # eval mode
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        y_pred += list(torch.argmax(logits,1).cpu().numpy())
        y_true += list(labels.cpu().numpy())

    meanf1score = f1_score(y_true, y_pred, average="micro")

    return meanf1score, y_true, y_pred


def plot_confusionmatrix(category_names, model, val_loader):

  metric_dict, y_true, y_pred = eval(model, val_loader)
  cf_matrix = confusion_matrix(y_true, y_pred)

  fig, ax = plt.subplots(figsize=(10,10))
  sns.heatmap(cf_matrix / cf_matrix.sum(axis=1).reshape(-1,1), annot=True, fmt='.2%', cmap='Blues', ax=ax)
  # labels, title and ticks
  ax.set_xlabel('Predicted labels');
  ax.set_ylabel('True labels');
  ax.set_title('Confusion Matrix');
  ax.xaxis.set_ticklabels(category_names, rotation = 45);
  ax.yaxis.set_ticklabels(category_names, rotation = 45);

  plt.show();


def test(model, train_transform, path):

    test_dataset = ImageFolder("../input/sapienza-training-camp-2022/test/", transform=train_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=False)
  
    y_pred = []
    model.eval()

    for images, labels in test_loader:
        images = images.to(device)
        
        with torch.no_grad():
            logits = model(images)
            
        y_pred += list(torch.argmax(logits,1).cpu().numpy())

    df_submission = pd.DataFrame()
    df_submission["file_name"] = [os.path.basename(p[0]) for p in test_dataset.samples]
    df_submission["category_id"] = y_pred
    df_submission.to_csv(path, index=False)
