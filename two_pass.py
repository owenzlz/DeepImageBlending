# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:28:28 2019

@author: Owen and Tarmily
"""

# Packages
import pdb
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imsave
from utils import *

# File Inputs
idx = 5
source_file = str(idx)+"_source.png"
mask_file = str(idx)+"_mask.png"
target_file = str(idx)+"_target.png"

# Hyperparameter Inputs
gpu_id = 0
num_steps = 1000
ss = 230; # source image size
ts = 512 # target image size
x_start = 382; y_start = 300 # blending location
grad_weight = 10000; style_weight = 10000; content_weight = 1; tv_weight = 10e-6; hist_weight = 0

# Load Images
source_img = np.array(Image.open(source_file).convert('RGB').resize((ss, ss)))
target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
mask_img = np.array(Image.open(mask_file).convert('L').resize((ss, ss)))
mask_img[mask_img>0] = 1

# Make Canvas Mask
canvas_mask = make_canvas_mask(x_start, y_start, target_img, mask_img)
canvas_mask = numpy2tensor(canvas_mask, gpu_id)
canvas_mask = canvas_mask.squeeze(0).repeat(3,1).view(3,ts,ts).unsqueeze(0)

# Compute Ground-Truth Gradients
gt_gradient = compute_gt_gradient(x_start, y_start, source_img, target_img, mask_img, gpu_id)

# Convert Numpy Images Into Tensors
source_img = torch.from_numpy(source_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
input_img = torch.randn(target_img.shape).to(gpu_id)

mask_img = numpy2tensor(mask_img, gpu_id)
mask_img = mask_img.squeeze(0).repeat(3,1).view(3,ss,ss).unsqueeze(0)

# Define LBFGS optimizer
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

optimizer = get_input_optimizer(input_img)

# Define Loss Functions
mse = torch.nn.MSELoss()

mean_shift = MeanShift(gpu_id)
vgg = Vgg16().to(gpu_id)


###################################
########### First Pass ###########
###################################

print('Optimizing...')
run = [0]
while run[0] <= num_steps:
    
    def closure():
        # Composite Foreground and Background to Make Blended Image
        blend_img = torch.zeros(target_img.shape).to(gpu_id)
        blend_img = input_img*canvas_mask + target_img*(canvas_mask-1)*(-1) 
        
        # Compute Laplacian Gradient of Blended Image
        pred_gradient = laplacian_filter_tensor(blend_img, gpu_id)
        
        # Compute Gradient Loss
        grad_loss = 0
        for c in range(len(pred_gradient)):
            grad_loss += mse(pred_gradient[c], gt_gradient[c])
        grad_loss /= len(pred_gradient)
        grad_loss *= grad_weight
        
        # Compute Style Loss
        target_features_style = vgg(mean_shift(target_img))
        target_gram_style = [gram_matrix(y) for y in target_features_style]
        
        blend_features_style = vgg(mean_shift(input_img))
        blend_gram_style = [gram_matrix(y) for y in blend_features_style]
        
        style_loss = 0
        for layer in range(len(blend_gram_style)):
            style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
        style_loss /= len(blend_gram_style)  
        style_loss *= style_weight           

        # Compute Content Loss
        blend_obj = blend_img[:,:,int(x_start-source_img.shape[2]*0.5):int(x_start+source_img.shape[2]*0.5), int(y_start-source_img.shape[3]*0.5):int(y_start+source_img.shape[3]*0.5)]
        source_object_features = vgg(mean_shift(source_img*mask_img))
        blend_object_features = vgg(mean_shift(blend_obj*mask_img))
        content_loss = content_weight * mse(blend_object_features.relu2_2, source_object_features.relu2_2)
        content_loss *= content_weight
        
        # Compute TV Reg Loss
        tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
                   torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
        tv_loss *= tv_weight
        
        # Compute Histogram Reg Loss
        hist_loss = 0

        match_time = time.process_time()

        if not hist_weight == 0:
            for layer in range(0, len(blend_features_style)):
                with torch.no_grad():
                    matched_features = get_matched_features_pytorch(blend_features_style[layer].detach(), target_features_style[layer])
                hist_loss += torch.norm(blend_features_style[layer] - matched_features, p="fro")

            match_time = time.process_time() - match_time

            hist_loss /= len(blend_features_style)
            hist_loss *= hist_weight
        else:
            hist_loss = torch.Tensor([0])
            match_time = 0

        
        # Compute Total Loss and Update Image
        if hist_weight == 0:
            loss = grad_loss + style_loss + content_loss + tv_loss
        else:
            loss = grad_loss + style_loss + content_loss + tv_loss + hist_loss
        optimizer.zero_grad()
        loss.backward()
        
        try:
            if run[0] % 1 == 0:
                blend_img_save = blend_img.clone()
                blend_img_save.data.clamp_(0, 255)
                blend_img_np = blend_img_save.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]
#                blend_img_np = np.clip(blend_img_np, 0, 255)
                imsave('process/first_'+str(run[0])+'.png', blend_img_np.astype(np.uint8))   
        except Exception: 
            pass
        
        # Print Loss
        if run[0] % 1 == 0:
            print("run {}:".format(run))
            print('grad : {:4f}, style : {:4f}, content: {:4f}, tv: {:4f}, hist: {:4f}, hist_time: {:6f}'.format(\
                          grad_loss.item(), \
                          style_loss.item(), \
                          content_loss.item(), \
                          tv_loss.item(), \
                          hist_loss.item(), \
                          match_time
                          ))
            print()
        
        run[0] += 1
        return loss
    
    optimizer.step(closure)

# clamp the pixels range into 0 ~ 255
input_img.data.clamp_(0, 255)

# Make the Final Blended Image
blend_img = torch.zeros(target_img.shape).to(gpu_id)
blend_img = input_img*canvas_mask + target_img*(canvas_mask-1)*(-1) 
blend_img_np = blend_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

imsave(str(idx)+'_first_pass.png', blend_img_np.astype(np.uint8))


###################################
########### Second Pass ###########
###################################

num_steps = 3000
style_weight = 10000000; content_weight = 1; tv_weight = 10e-6
ss = 512; ts = 512

first_pass_img_file = str(idx)+'_first_pass.png'

first_pass_img = np.array(Image.open(first_pass_img_file).convert('RGB').resize((ss, ss)))
target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
first_pass_img = torch.from_numpy(first_pass_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)

# Define LBFGS optimizer
def get_input_optimizer(first_pass_img):
    optimizer = optim.LBFGS([first_pass_img.requires_grad_()])
    return optimizer

optimizer = get_input_optimizer(first_pass_img)

print('Optimizing...')
run = [0]
while run[0] <= num_steps:
    
    def closure():
        
        # Compute Loss Loss    
        target_features_style = vgg(mean_shift(target_img))
        target_gram_style = [gram_matrix(y) for y in target_features_style]
        blend_features_style = vgg(mean_shift(first_pass_img))
        blend_gram_style = [gram_matrix(y) for y in blend_features_style]
        style_loss = 0
        for layer in range(len(blend_gram_style)):
            style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
        style_loss /= len(blend_gram_style)  
        style_loss *= style_weight        
        
        # Compute Content Loss
        content_features = vgg(mean_shift(first_pass_img))
        content_loss = content_weight * mse(blend_features_style.relu2_2, content_features.relu2_2)

        # Compute TV Reg Loss
        tv_loss = torch.sum(torch.abs(first_pass_img[:, :, :, :-1] - first_pass_img[:, :, :, 1:])) + \
                   torch.sum(torch.abs(first_pass_img[:, :, :-1, :] - first_pass_img[:, :, 1:, :]))
        tv_loss *= tv_weight
        
        # Compute Total Loss and Update Image
        loss = style_loss + content_loss + tv_loss
        optimizer.zero_grad()
        loss.backward()
        
        try:
            if run[0] % 10 == 0:
                first_pass_img_save = first_pass_img.clone()
                first_pass_img_save.data.clamp_(0, 255)
                first_pass_img_save_np = first_pass_img_save.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]
                imsave('process/second_'+str(run[0])+'.png', first_pass_img_save_np.astype(np.uint8))   
        except Exception: 
            pass
        
        # Print Loss
        if run[0] % 1 == 0:
            print("run {}:".format(run))
            print(' style : {:4f}, content: {:4f}'.format(\
                          style_loss.item(), \
                          content_loss.item()
                          ))
            print()
        
        run[0] += 1
        return loss
    
    optimizer.step(closure)

# clamp the pixels range into 0 ~ 255
first_pass_img.data.clamp_(0, 255)

# Make the Final Blended Image
input_img_np = first_pass_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

imsave(str(idx)+'_second_pass.png', input_img_np.astype(np.uint8))










