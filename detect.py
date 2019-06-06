import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import argparse
import sys
import pickle
import time

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 

    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]

    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)

    return img_, dim


def arg_parse(): 
	parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
	parser.add_argument("--image", dest = 'image', help = 
						"image input for classification",
						default = "imgs/Malamute.jpg", type = str)
	parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.20)
	parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file",
						default = "cfg/tiny.cfg", type = str)
	parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile",
						default = "weights/tiny.weights", type = str)

	parser.add_argument("--names", dest = 'names_list', help = "names-class file",
						default = "data/imagenet.shortnames.list", type = str)
	
	return parser.parse_args()

def get_top_k(frame, CUDA, k = 5):

	img, dim = prep_image(frame, int(model.net_info["height"]))

	if CUDA:
		img = img.cuda()

	with torch.no_grad():   
		output = model(Variable(img), CUDA)

	max_conf_k, max_conf_index_k = torch.topk(output,k) 

	return max_conf_k, max_conf_index_k 

def load_names(fname,num_classes):

	arr_lines = []

	with open(fname,"r") as f:
		lines = f.readlines()
		arr_lines = [line.strip() for line in lines[:num_classes]]

	return arr_lines


if __name__ == '__main__':

	args = arg_parse()

	img_path = args.image
	names_file_path = args.names_list

	arr_names = load_names(names_file_path,1000)

	CUDA = torch.cuda.is_available()

	print("Loading network.....")

	model = Darknet(args.cfgfile)
	model.load_weights(args.weightsfile)

	print("Network successfully loaded")

	inp_dim = int(model.net_info["height"])
	CUDA = torch.cuda.is_available()
	
	if CUDA:
	    model.cuda()

	model.eval()	### Freeze weights. Use model for inferencing

	### Start CLassification ###

	frame = cv2.imread(img_path)

	start = time.time()

	max_conf_k, max_conf_index_k  = get_top_k(frame,CUDA)

	max_conf_k = max_conf_k.squeeze()
	max_conf_index_k = max_conf_index_k.squeeze() 				### Batch size = 1. Change for multiple batches

	total_time = time.time() - start
	
	# print("Total inference time : {}".format(total_time))

	for conf, idx in zip(max_conf_k,max_conf_index_k):
		name = arr_names[int(idx)]
		print("Class = {}  Confidence = {}".format(name,conf*100))
