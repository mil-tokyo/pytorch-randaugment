# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image
import inspect
from pixelchangenet import *
import torchvision.transforms as transforms
def ShearX(value):
    weight_base = torch.eye(2,dtype=torch.float32)
    bias_base = torch.zeros(2,dtype=torch.float32)
    weight_random = torch.tensor([[[0,value],[0,0]]],dtype=torch.float32)
    bias_random = torch.tensor([[-0.5*value,0]],dtype=torch.float32)
    return RandomAffineNet(weight_base,bias_base,weight_random,bias_random)
def ShearY(value):
    weight_base = torch.eye(2,dtype=torch.float32)
    bias_base = torch.zeros(2,dtype=torch.float32)
    weight_random = torch.tensor([[[0,0],[value,0]]],dtype=torch.float32)
    bias_random = torch.tensor([[0,-0.5*value]],dtype=torch.float32)
    return RandomAffineNet(weight_base,bias_base,weight_random,bias_random)
    
def TranslateX(value):
    return TranslateXNet(value)

def TranslateY(value):
    return TranslateYNet(value)

def Rotate(value):
    return RandomRotateNet(value)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [0, 8]
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v, fillcolor):  # [0, 60] => percentage: [0, 0.2]
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v,fillcolor = fillcolor)


def CutoutAbs(img, v, fillcolor):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = random.uniform(0,w)
    y0 = random.uniform(0,h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, fillcolor)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = random.randrange(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


class Identity(nn.Module):
    def __init__(self,value):
        super(Identity,self).__init__()
    def forward(self,x):
        return x

def augment_list(is_smallimage):  # 16 oeprations and their ranges
    if is_smallimage:
        #https://github.com/tensorflow/models/blob/master/research/autoaugment/augmentation_transforms.py
        l = [
            (Identity, 0., 1.0,True),
            (ShearX, 0., 0.3,True),  # 0
            (ShearY, 0., 0.3,True),  # 1
            (TranslateX, 0., 0.33,True),  # 2
            (TranslateY, 0., 0.33,True),  # 3
            (Rotate, 0, 30,True),  # 4
            (AutoContrast, 0, 1,False),  # 5
            #(Invert, 0, 1),  # 6
            (Equalize, 0, 1,False),  # 7
            #(Solarize, 0, 256,False),  # 8
            (Posterize, 0, 4,False),  # 9
            (Contrast, 0.1, 1.0,False),  # 10
            (Color, 0.1, 1.0,False),  # 11
            (Brightness, 0.1, 1.0,False),  # 12
            (Sharpness, 0.1, 1.0,False),  # 13
            # (Cutout, 0, 0.2),  # 14
            # (SamplePairing(imgs), 0, 0.4),  # 15
        ]
    else:
        # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
        l = [
            (AutoContrast, 0, 1),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Rotate, 0, 30),
            (Posterize, 0, 4),
            (Solarize, 0, 256),
            (SolarizeAdd, 0, 110),
            (Color, 0.1, 1.9),
            (Contrast, 0.1, 1.9),
            (Brightness, 0.1, 1.9),
            (Sharpness, 0.1, 1.9),
            (ShearX, 0., 0.3),
            (ShearY, 0., 0.3),
            (CutoutAbs, 0, 40),
            (TranslateXabs, 0., 100),
            (TranslateYabs, 0., 100),
        ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = random.randrange(h)
        x = random.randrange(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img*mask


class RandAugment:
    def __init__(self, n, m, is_smallimage, fillcolor = (0,0,0)):
        self.n = n
        self.m = m      # [0, 30]
        self.fillcolor = fillcolor
        self.augment_list = augment_list(is_smallimage)

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        ops_nonmatrix = []
        ops_matrix = []
        for op, minval, maxval,ismatrix in ops:
            if ismatrix == False:
                ops_nonmatrix.append((op,minval,maxval))
            else:
                ops_matrix.append(op((float(self.m)/10)*float(maxval-minval)+minval))
        img_moved = transforms.ToPILImage()(PixelChangeAugment(nn.Sequential(*ops_matrix))(img))
        for op,minval,maxval in ops_nonmatrix:
            val = (float(self.m) / 10) * float(maxval - minval) + minval
            if 'fillcolor' in inspect.getfullargspec(op)[0]:
              assert 'fillcolor' == inspect.getfullargspec(op)[0][-1]
              img_moved = op(img_moved, val, self.fillcolor)
            else:
              img_moved = op(img_moved, val)

        return transforms.ToTensor()(img_moved)
