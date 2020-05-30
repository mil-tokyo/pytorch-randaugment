import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import random

class RandomAffineNet(nn.Module):
    def __init__(self,weight_base,bias_base,weight_comp_list,bias_comp_list):
        super(RandomAffineNet,self).__init__()
        self.weight_base = weight_base
        self.bias_base = bias_base
        self.weight_comp_list = weight_comp_list
        self.bias_comp_list = bias_comp_list
    def set_comp_list(self,weight_base,bias_base,weight_comp_list,bias_comp_list):#input: tensor
        self.weight_base = weight_base
        self.bias_base = bias_base
        self.weight_comp_list = weight_comp_list
        self.bias_comp_list = bias_comp_list
    def forward(self,x):
        rand = np.array([random.gauss(0,1) for i in range(self.weight_comp_list.shape[0])])
        bias_rand = torch.tensor(rand.reshape(-1,1))
        weight_rand = torch.tensor(rand.reshape(-1,1,1))
        weight = self.weight_base+torch.sum(weight_rand * self.weight_comp_list,dim=0,dtype=torch.float32)
        bias = self.bias_base + torch.sum(bias_rand * self.bias_comp_list,dim=0,dtype=torch.float32)
        return F.linear(x,weight,bias)
class TranslateXNet(nn.Module):
  def __init__(self,magnitude_transform):
    super(TranslateXNet,self).__init__()
    self.magnitude_transform = magnitude_transform
  def forward(self,x):
    retval = x - torch.tensor([0,random.uniform(-1,1)*self.magnitude_transform],dtype=torch.float32).reshape(1,2)
    return retval
class TranslateYNet(nn.Module):
  def __init__(self,magnitude_transform):
    super(TranslateYNet,self).__init__()
    self.magnitude_transform = magnitude_transform
  def forward(self,x):
    retval = x - torch.tensor([random.uniform(-1,1)*self.magnitude_transform,0],dtype=torch.float32).reshape(1,2)
    return retval
class RandomFlipNet(nn.Module):
  def __init__(self):
    super(RandomFlipNet,self).__init__()
  def forward(self,x):
    retval = x
    if random.random()>0.5:
      #flip the image
      retval[:,1]=1-retval[:,1]
    return retval
class RandomRotateNet(nn.Module):
  def __init__(self,degree):
    super(RandomRotateNet,self).__init__()
    DEGREE_TO_RADIAN = 3.1415 / 180
    self.degree = degree * DEGREE_TO_RADIAN
  def forward(self,x):
    t_degree = random.uniform(-self.degree,self.degree)
    weight = torch.tensor([[np.cos(t_degree),np.sin(t_degree)],[-np.sin(t_degree),np.cos(t_degree)]],dtype=torch.float32)
    bias = torch.tensor([0.5,0.5],dtype=torch.float32)
    return F.linear(x-bias,weight,bias)
class CutoutNet(nn.Module):
  def __init__(self,v):
    super(CutoutNet,self).__init__()
    self.v = v
  def forward(self,x):
    x0 = random.uniform(0,1)
    y0 = random.uniform(0,1)
    x0 = max(0,x0-self.v/2.0)
    y0 = max(0,y0-self.v/2.0)
    x1 = min(1,x0+self.v)
    y1 = min(1,y0+self.v)
    mask = (x0<x[:,0])&(x[:,0]<x1)&(y0<x[:,1])&(x[:,1]<y1)
    x[mask]=np.nan
    return x

class PixelChangeAugment(object):
    def __init__(self,net,fillpixel = None):#net: pytorch network which calculates the correspondence with input and output
        self.net = net
        self.fillpixel = fillpixel
    def __call__(self,img):#input:tensor
        _,H,W = img.shape
        if self.fillpixel is None:
            average_pixel = torch.tensor([0.4914,0.4824,0.4466],dtype=torch.float32)
        else:
            average_pixel = self.fillpixel
        #prepare input(cpu) to PixelNet begin
        H_linspace = torch.linspace(0,1,steps=H,dtype=torch.float32)
        W_linspace = torch.linspace(0,1,steps=W,dtype=torch.float32)
        grid_H,grid_W = torch.meshgrid([H_linspace,W_linspace])
        pixel_input = torch.stack([grid_H,grid_W],dim=2).reshape(H*W,2)
        #prepare input to PixelNet end
        #calculate PixelNet begin
        target_pixel = self.net(pixel_input).reshape(H,W,2)
        target_pixel[:,:,0]*=(H-1)
        target_pixel[:,:,1]*=(W-1)
        #calculate PixelNet end
        #deal with wrong output
        proper_pixel = (target_pixel[:,:,0]<H-1 )&(target_pixel[:,:,0]>=0)&(target_pixel[:,:,1]>=0)&(target_pixel[:,:,1]<W-1)
        retval = torch.zeros_like(img,dtype=torch.float32)
        temp_input = torch.tensor([H/2,W/2],dtype=torch.float32)
        target_pixel[~proper_pixel] = temp_input.reshape(1,-1)
        #upper-left
        upper_left_index= torch.floor(target_pixel).type(torch.int64)
        retval += img[:,upper_left_index[:,:,0],upper_left_index[:,:,1]] * (1+upper_left_index[:,:,0]-target_pixel[:,:,0])*(1+upper_left_index[:,:,1]-target_pixel[:,:,1])
        retval += img[:,upper_left_index[:,:,0]+1,upper_left_index[:,:,1]] * (target_pixel[:,:,0]-upper_left_index[:,:,0])*(1+upper_left_index[:,:,1]-target_pixel[:,:,1])
        retval += img[:,upper_left_index[:,:,0],upper_left_index[:,:,1]+1] * (1+upper_left_index[:,:,0]-target_pixel[:,:,0])*(-upper_left_index[:,:,1]+target_pixel[:,:,1])
        retval += img[:,upper_left_index[:,:,0]+1,upper_left_index[:,:,1]+1] * (-upper_left_index[:,:,0]+target_pixel[:,:,0])*(-upper_left_index[:,:,1]+target_pixel[:,:,1])
        retval *= proper_pixel.reshape(1,H,W)
        retval += (~proper_pixel.reshape(1,H,W))*average_pixel.reshape(-1,1,1)
        return retval

