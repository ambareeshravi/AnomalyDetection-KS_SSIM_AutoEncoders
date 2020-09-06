import torch
import os

class Config:
	# General Configuration
	Cuda = torch.cuda.is_available()
	ngpu = 0
	if Cuda: ngpu = 1
	device = torch.device("cuda:0" if Cuda else "cpu")
	print("-- Running on", device, "--")

	# Model configuration
	ImageSize = 128
	channels = 3
	EmbeddingSize = 300
	save_path = 'models/'