# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:28:28 2019

@author: Owen
"""
from torch.nn import functional as F
import torch.nn as nn
from PIL import Image 
import numpy as np
from skimage.io import imsave
import cv2
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision import models
from collections import namedtuple
import pdb
import copy

def numpy2tensor(np_array, gpu_id):
    if len(np_array.shape) == 2:
        tensor = torch.from_numpy(np_array).unsqueeze(0).float().to(gpu_id)
    else:
        tensor = torch.from_numpy(np_array).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
    return tensor

def make_canvas_mask(x_start, y_start, target_img, mask):
    canvas_mask = np.zeros((target_img.shape[0], target_img.shape[1]))
    canvas_mask[int(x_start-mask.shape[0]*0.5):int(x_start+mask.shape[0]*0.5), int(y_start-mask.shape[1]*0.5):int(y_start+mask.shape[1]*0.5)] = mask
    return canvas_mask

def laplacian_filter_tensor(img_tensor, gpu_id):

    laplacian_filter = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
    laplacian_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    laplacian_conv.weight = nn.Parameter(torch.from_numpy(laplacian_filter).float().unsqueeze(0).unsqueeze(0).to(gpu_id))
    
    for param in laplacian_conv.parameters():
        param.requires_grad = False
    
    red_img_tensor = img_tensor[:,0,:,:].unsqueeze(1)
    green_img_tensor = img_tensor[:,1,:,:].unsqueeze(1)
    blue_img_tensor = img_tensor[:,2,:,:].unsqueeze(1)
    
    red_gradient_tensor = laplacian_conv(red_img_tensor).squeeze(1) 
    green_gradient_tensor = laplacian_conv(green_img_tensor).squeeze(1) 
    blue_gradient_tensor = laplacian_conv(blue_img_tensor).squeeze(1)
    return red_gradient_tensor, green_gradient_tensor, blue_gradient_tensor

def compute_gt_gradient(x_start, y_start, source_img, target_img, mask, gpu_id):
    
    # compute source image gradient
    source_img_tensor = torch.from_numpy(source_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
    red_source_gradient_tensor, green_source_gradient_tensor, blue_source_gradient_tenosr = laplacian_filter_tensor(source_img_tensor, gpu_id)    
    red_source_gradient = red_source_gradient_tensor.cpu().data.numpy()[0]
    green_source_gradient = green_source_gradient_tensor.cpu().data.numpy()[0]
    blue_source_gradient = blue_source_gradient_tenosr.cpu().data.numpy()[0]
    
    # compute target image gradient
    target_img_tensor = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
    red_target_gradient_tensor, green_target_gradient_tensor, blue_target_gradient_tenosr = laplacian_filter_tensor(target_img_tensor, gpu_id)    
    red_target_gradient = red_target_gradient_tensor.cpu().data.numpy()[0]
    green_target_gradient = green_target_gradient_tensor.cpu().data.numpy()[0]
    blue_target_gradient = blue_target_gradient_tenosr.cpu().data.numpy()[0]    
    
    # mask and canvas mask
    canvas_mask = np.zeros((target_img.shape[0], target_img.shape[1]))
    canvas_mask[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = mask
    
    # foreground gradient
    red_source_gradient = red_source_gradient * mask
    green_source_gradient = green_source_gradient * mask
    blue_source_gradient = blue_source_gradient * mask
    red_foreground_gradient = np.zeros((canvas_mask.shape))
    red_foreground_gradient[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = red_source_gradient
    green_foreground_gradient = np.zeros((canvas_mask.shape))
    green_foreground_gradient[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = green_source_gradient
    blue_foreground_gradient = np.zeros((canvas_mask.shape))
    blue_foreground_gradient[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = blue_source_gradient
    
    # background gradient
    red_background_gradient = red_target_gradient * (canvas_mask - 1) * (-1)
    green_background_gradient = green_target_gradient * (canvas_mask - 1) * (-1)
    blue_background_gradient = blue_target_gradient * (canvas_mask - 1) * (-1)
    
    # add up foreground and background gradient
    gt_red_gradient = red_foreground_gradient + red_background_gradient
    gt_green_gradient = green_foreground_gradient + green_background_gradient
    gt_blue_gradient = blue_foreground_gradient + blue_background_gradient
    
    gt_red_gradient = numpy2tensor(gt_red_gradient, gpu_id)
    gt_green_gradient = numpy2tensor(gt_green_gradient, gpu_id)  
    gt_blue_gradient = numpy2tensor(gt_blue_gradient, gpu_id)
    
    gt_gradient = [gt_red_gradient, gt_green_gradient, gt_blue_gradient]
    return gt_gradient

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

class MeanShift(nn.Conv2d):
    def __init__(self, gpu_id):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        rgb_range=1
        rgb_mean=(0.4488, 0.4371, 0.4040)
        rgb_std=(1.0, 1.0, 1.0)
        sign=-1
        std = torch.Tensor(rgb_std).to(gpu_id)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1).to(gpu_id) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean).to(gpu_id) / std
        for p in self.parameters():
            p.requires_grad = False


