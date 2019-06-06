
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
# from bbox import bbox_iou
import time

import GPUtil

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix


def get_offset(idx,num_anchors,grid_size):

    idx_test = idx[:,1] - idx[:,1]%num_anchors   ### Subtract anchor num offset
    x_offset = (idx_test/num_anchors)%grid_size
    idx_test_1 = idx_test/num_anchors - x_offset
    y_offset = idx_test_1/grid_size

    return x_offset.unsqueeze(1),y_offset.unsqueeze(1)

def advanced_indexing(tensor, index):
    if isinstance(index, tuple):
        adv_loc = []
        for i, el in enumerate(index):
            if isinstance(el, torch.LongTensor):
                adv_loc.append((i, el))
        if len(adv_loc) < 2:
            return tensor[index]
        
        # check that number of elements in each indexing array is the same
        len_array = [i.numel() for _, i in adv_loc]
        #assert len_array.count(len_array[0]) == len(len_array)
        
        idx = [i for i,_ in adv_loc]
        sizes = [tensor.size(i) for i in idx]
        new_size = [tensor.size(i) for i in range(tensor.dim()) if i not in idx]
        new_size_final = [tensor.size(i) for i in range(tensor.dim()) if i not in idx]

        start_idx = idx[0]
        # if there is a space between the indexes
        if idx[-1] - idx[0] + 1 != len(idx):
            permute = idx + [i for i in range(tensor.dim()) if i not in idx]
            tensor = tensor.permute(*permute).contiguous()
            start_idx = 0
        
        lin_idx = _linear_index(sizes, [i for _, i in adv_loc])
        reduc_size = reduce(mul, sizes)
        new_size.insert(start_idx, reduc_size)
        new_size_final[start_idx:start_idx] = list(adv_loc[0][1].size())

        tensor = tensor.view(*new_size)
        tensor = tensor.index_select(start_idx, lin_idx)
        tensor = tensor.view(new_size_final)
        
        return tensor
        
    else:
        return tensor[index]

def filter_boxes_gpu(prediction,conf = 0.2):

    non_zero_ind =  torch.gt(prediction[:,:,4],conf)
    idx = torch.nonzero(non_zero_ind)
    # prediction = prediction[:,idx[:,1],:]

    prediction = advanced_indexing(prediction, non_zero_ind).unsqueeze(0)

    print(prediction.shape)

    return non_zero_ind,idx,prediction

def filter_boxes_cpu(prediction,conf = 0.2):

    prediction = prediction.cpu().numpy()       ### Np Array

    non_zero_ind = prediction[:,:,4] > conf       ### Np Filtering
    idx = np.asarray(non_zero_ind.nonzero())
    prediction = prediction[non_zero_ind]

    ### To GPU ###

    prediction = torch.from_numpy(prediction)
    prediction = prediction.to(0).unsqueeze(0)

    idx = torch.from_numpy(idx)
    idx = idx.to(0).transpose(0,1)

    return non_zero_ind,idx,prediction


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    # num_anchors = dict_mesh[stride]["num_anchors"]

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)


    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    prediction_orig = prediction

    non_zero_ind_cpu,idx_cpu,prediction_cpu = filter_boxes_gpu(prediction_orig)

    x_offset, y_offset = get_offset(idx_cpu,num_anchors,grid_size)

    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

    idx_anchors = idx_cpu[:,1]%3

    idx_test_anchor = idx_cpu[:,1].squeeze()

    anchors = anchors[:,idx_test_anchor]

    x_offset = x_offset.type(torch.cuda.FloatTensor)
    y_offset = y_offset.type(torch.cuda.FloatTensor)

    x_y_offset = torch.cat((x_offset, y_offset), 1).view(-1,2).unsqueeze(0)

    prediction_cpu[:,:,:2] += x_y_offset

    prediction_cpu[:,:,2:4] = torch.exp(prediction_cpu[:,:,2:4])*anchors

    #Softmax the class scores
    prediction_cpu[:,:,5: 5 + num_classes] = torch.sigmoid((prediction_cpu[:,:, 5 : 5 + num_classes]))

    prediction_cpu[:,:,:4] *= stride
    
    return prediction_cpu

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def get_im_dim(im):
    im = cv2.imread(im)
    w,h = im.shape[1], im.shape[0]
    return w,h

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res