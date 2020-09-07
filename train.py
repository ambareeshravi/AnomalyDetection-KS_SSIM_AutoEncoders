import os, cv2, shutil
import pickle as pkl, json
import numpy as np
import pandas as pd

from time import time
from tqdm import tqdm
from glob import glob

import torch
from torch import nn
from PIL import Image

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import gc

from config import Config
from models import *
from data import *
from losses import *
from utils import *
from weights_init import weights_init

np.random.seed(0)
torch.manual_seed(0)

class Trainer:
	'''
	Trains the model according to the given parameters
	'''
	def __init__(self):
		pass

	def plot_history(self, history, model_type, dataset_type):
		'''
		Plots graph with model history

		Args:
			history -  dict with parameter as key and list of values
			model_type - model name as <str>
			dataset_type - dataset name as <str>
		Returns:
			-
		Exception:
			general exception
		'''
		try:
			epochs = list(range(len(history["train_loss"])))
			plt.plot(epochs, history["train_loss"], label = "Train Loss")
			plt.plot(epochs, history["val_loss"], label = "Validation Loss")
			plt.legend()
			plt.xlabel("Number of epochs")
			plt.ylabel("Loss")
			plt.title("Performance Curves of %s on %s dataset"%(model_type, dataset_type))
			plt.savefig(os.path.join(self.save_folder, "train_history.png"), dpi = 100, bbox_inches='tight')
			plt.clf()
			x = plt.imread(os.path.join(self.save_folder, "train_history.png"))
			plt.axis('off')
			plt.imshow(x)
		except Exception as e:
			print(e)
		
	def step(self, images, isTrain = True):
		images = images.to(Config.device)
		self.optimizer.zero_grad()
		if isTrain: self.model.train()
		else: self.model.eval()
		
		if self.isVariational:
			reconstruction, mu, logvar = self.model(images)
			loss, components = self.loss_function(images, reconstruction, mu, logvar)
# 			print(components)
		elif self.isContractive:
			reconstruction, encoding = self.model(images)
			loss, components = self.loss_function(images, reconstruction, encoding, self.model.encoder[-1][0].weight)
# 			print(components)
		else:
			reconstruction, encoding = self.model(images)
			loss = self.loss_function(images, reconstruction)
			
		if isTrain:
			loss.backward()
			self.optimizer.step()
		return loss.item()

	def train(self, model_type,
			pretrained_model = False,
			dataset_type = "HAM10000",
			lr = 1e-4,
			epochs = 200,
			batch_size = 32,
			starting_epoch = 1,
			version = 1,
			decayBy = 0.85,
			decayPer = 20,
			init_weights = True,
			debug = True):
		'''
		Trains the models

		Args:
			model_type - model name as <str>
			dataset_type - dataset name as <str>
			pretrained_model - path to any pretrained model as <str>
		Returns:
			-
		Exception:
			-
		'''
		self.model_name, self.save_folder = create_path(model_type, dataset_type, version, not pretrained_model)
		
		# Prepare dataset
		ds = selectData(dataset_type, batch_size = batch_size)
		train_dataset, train_dataloader, _, val_dataloader = ds.getTrainData()
		if debug: print("Data Loaded")

		# Prepare model
		self.isVariational, self.isContractive = False, False
		if "variational" in model_type.lower(): self.isVariational = True
		if "contractive" in model_type.lower(): self.isContractive = True
		self.model, self.loss_function = selectModel(model_type)
		if init_weights: self.model.apply(weights_init)
		if pretrained_model: load_model(self.model, pretrained_model)
		self.model.to(Config.device)
		if debug: print("Model Ready")

		# Define optimizer and Loss function
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, betas=(0.5, 0.999))
		
		# Create variables to record history and store the best models
		history = {"train_loss" : list(), "val_loss": list()}
		
		# Start training
		for epoch in range(starting_epoch, epochs + starting_epoch):
			batch_st = time()
			train_loss, validation_loss = list(), list()
			
			# Train
			for batch_idx, data in enumerate(train_dataloader, 0):
				images, labels = data
				train_loss.append(self.step(images))
				
			# validate
			for val_batch_idx, batch_val_data in enumerate(val_dataloader):
				with torch.no_grad():
					val_images, val_labels = batch_val_data
					validation_loss.append(self.step(val_images, isTrain = False))
			
			# Calculate average loss					
			train_loss = np.mean(train_loss)
			validation_loss = np.mean(validation_loss)
			
			# Record to history
			history["train_loss"].append(train_loss)
			history["val_loss"].append(validation_loss)
			
			# Learning Rate Decay
			if decayPer and (epoch%decayPer == 0): adjust_learning_rate(self.optimizer, decayBy)
			
			# Print Status of the Epoch
			print("Epoch: %03d / %03d \nTraining: LOSS: %.4f | Validation: LOSS: %0.4f | time/epoch: %d seconds " % (epoch, epochs + starting_epoch - 1, train_loss, validation_loss, time() - batch_st))

			save_model(self.model, os.path.join(self.save_folder, self.model_name))

		# Save final model
		pkl.dump(history, open(os.path.join(self.save_folder, self.model_name.split(".")[0]+"_history.pkl"), "wb"))
		self.plot_history(history, model_type, dataset_type)
		try:
			del self.model, ds, self.optimizer
			torch.cuda.empty_cache()
			gc.collect()
		except Exception as e:
			print("Could not clear the memory. Kill the process manually.")
			print(e)
			
		return self.save_folder
    
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_type', required=True, default='Base', help='Type of the model to train')
	parser.add_argument('--dataset_type', required=True, default='HAM10000', help='Type of the dataset to train on')
	args = parser.parse_args()
	
	tr = Trainer()
	tr.train(model_type = args.model_type, dataset_type = args.dataset_type)
	models_path = tr.save_folder
	
	del tr
	gc.collect()
	
	from test_models import *
	# Test
	ts = Tester()
	for model_path in glob(os.path.join(models_path, "*.tar")):    
		try:
			print("Testing:", model_path)
			ts.test(model_path)
			print("-"*20)
		except:
			pass
	del ts
	gc.collect()