import torch
import torchvision
import pandas as pd
from PIL import Image
import os
class CustomImagenet(torchvision.datasets.VisionDataset):
  def __init__(self,root,split,transforms=None,transform=None,target_transform = None):
    super(CustomImagenet,self).__init__(root,transforms,transform,target_transform)
    assert split == 'train' or split == 'val' or split == 'test'
    self.split = split
    self.index_file = pd.read_csv(os.path.join(self.root,split+'.txt'),sep=' ',header = None)
  def __len__(self):
    return len(self.index_file)
  def __getitem__(self,x):
    image_path = os.path.join(self.root,'CLS-LOC',self.split,self.index_file.iat[x,0])
    with open(image_path, 'rb') as f:
      img = Image.open(f)
      pil_image = img.convert('RGB')

    target = self.index_file.iat[x,1]
    if self.transforms is None:
      return (pil_image,target)
    return self.transforms(pil_image,target)
    

