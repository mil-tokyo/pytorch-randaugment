import torch
import torchvision
import pandas as pd
from PIL import Image
import os
import numpy as np
class CustomImagenet(torchvision.datasets.VisionDataset):
  def __init__(self,root,split,transforms=None,transform=None,target_transform = None):
    super(CustomImagenet,self).__init__(root,transforms,transform,target_transform)
    assert split == 'train' or split == 'val' or split == 'test'
    self.split = split
    index_file = pd.read_csv(os.path.join(self.root,split+'.txt'),sep=' ',header = None)
    self.file_path = np.array(index_file[0].values,dtype=np.str)
    self.target_index = np.array(index_file[1].values,dtype=np.int64)
  def __len__(self):
    return len(self.target_index)
  def __getitem__(self,x):
    image_path = os.path.join(self.root,'CLS-LOC',self.split,self.file_path[x])
    with open(image_path, 'rb') as f:
      img = Image.open(f)
      pil_image = img.convert('RGB')

    target = self.target_index[x]
    if self.transforms is None:
      return (pil_image,target)
    return self.transforms(pil_image,target)
    

