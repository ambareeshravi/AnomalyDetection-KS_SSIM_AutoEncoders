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

	def plot_history(self, history, model_type, dataset_type, save_path):
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
			plt.savefig(os.path.join(save_path, "train_history.png"), dpi = 100, bbox_inches='tight')
			plt.clf()
			x = plt.imread(os.path.join(save_path, "train_history.png"))
			plt.axis('off')
			plt.imshow(x)
		except Exception as e:
			print(e)

	def setup_model(self, model_type, dataset_type, lr = None, isPretrained = False):
		model_name, save_folder = create_path(model_type, dataset_type, self.version, not isPretrained)
		model, loss_function = selectModel(model_type)
		if self.init_weights: model.apply(weights_init)
		optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.5, 0.999))
		if self.debug: print(model_type, "setup ready")
		isContractive, isVariational = False, False
		if "contractive" in model_type.lower(): isContractive = True
		if "variational" in model_type.lower(): isVariational = True
		return model_name, save_folder, model, optimizer, loss_function, isContractive, isVariational
	
	def epoch_status(self, epoch, batch_st, batch_end):
		print("Epoch: %03d / %03d | time/epoch: %d seconds"%(epoch, self.total_epochs, (batch_end - batch_st)))
		print("-"*60)
		for model_type in self.models_info.keys():
			if self.models_info[model_type]["stop"]: continue
			print("Model:", model_type)
			if epoch > (self.starting_epoch + 2):
				d = {
				"Training" : {"Loss -2": self.models_info[model_type]["history"]["train_loss"][-3],
							"Loss -1": self.models_info[model_type]["history"]["train_loss"][-2],
							"Loss" : self.models_info[model_type]["history"]["train_loss"][-1],
							 },
				"Validation" : {"Loss -2": self.models_info[model_type]["history"]["val_loss"][-3],
								"Loss -1": self.models_info[model_type]["history"]["val_loss"][-2],
								"Loss" : self.models_info[model_type]["history"]["val_loss"][-1],
							   },
			}
			else:
				d = {
				"Training" : {"Loss" : self.models_info[model_type]["history"]["train_loss"][-1],
							 },
				"Validation" : {"Loss" : self.models_info[model_type]["history"]["val_loss"][-1],
							   },
				}
			print(pd.DataFrame(d).T)
			print("-"*40)
			
	def epoch_reset(self):
		for model_type in self.models_info.keys():
			if self.models_info[model_type]["stop"]: continue
			self.models_info[model_type]["train_loss"] = list()
			self.models_info[model_type]["val_loss"] = list()

	def save_models(self, dataset_type, final = False):
		for model_type in self.models_info.keys():
			save_model(self.models_info[model_type]["model"], os.path.join(self.models_info[model_type]["save_path"], self.models_info[model_type]["model_name"]))
			if final:
				pkl.dump(self.models_info[model_type]["history"], open(os.path.join(self.models_info[model_type]["save_path"], self.models_info[model_type]["model_name"].split(".")[0]+"_history.pkl"), "wb"))
				self.plot_history(self.models_info[model_type]["history"], model_type, dataset_type, self.models_info[model_type]["save_path"])
		
	def step(self, model, optimizer, loss_function, images, isVariational, isContractive, isTrain = True):
		model.to(Config.device)
		images = images.to(Config.device)
		optimizer.zero_grad()
		if isTrain: model.train()
		else: model.eval()
		
		if isVariational:
			reconstruction, mu, logvar = model(images)
			loss, bce, kld = loss_function(images, reconstruction, mu, logvar)
		elif isContractive:
			reconstruction, encoding = model(images)
			loss = loss_function(images, reconstruction, encoding, model.encoder[-1][0].weight)
		else:
			reconstruction, encoding = model(images)
			loss = loss_function(images, reconstruction)
			
		if isTrain:
			loss.backward()
			optimizer.step()
			
		model.to('cpu')
		return loss.item()
	
	def step_normal(self, model, optimizer, loss_function, images, isTrain = True):
		model.to(Config.device)
		images = images.to(Config.device)
		optimizer.zero_grad()
		if isTrain: model.train()
		else: model.eval()

		reconstruction, encoding = model(images)
		loss = loss_function(images, reconstruction)
			
		if isTrain:
			loss.backward()
			optimizer.step()
			
		model.to('cpu')
		return loss.item()
	
	def step_contractive(self, model, optimizer, loss_function, images, isTrain = True):
		model.to(Config.device)
		images = images.to(Config.device)
		optimizer.zero_grad()
		if isTrain: model.train()
		else: model.eval()
		
		reconstruction, encoding = model(images)
		loss, components = loss_function(images, reconstruction, encoding, model.encoder[-1][0].weight)
		if isTrain:
			loss.backward()
			optimizer.step()
			
		model.to('cpu')
		return loss.item()
	
	def step_variational(self, model, optimizer, loss_function, images, isTrain = True):
		model.to(Config.device)
		images = images.to(Config.device)
		optimizer.zero_grad()
		if isTrain: model.train()
		else: model.eval()
			
		reconstruction, mu, logvar = model(images)
		loss, components = loss_function(images, reconstruction, mu, logvar)
			
		if isTrain:
			loss.backward()
			optimizer.step()
			
		model.to('cpu')
		return loss.item()

	
	def lr_decay(self, epoch):
		for model_type in self.models_info.keys():
			if self.models_info[model_type]["stop"]: continue
			if self.models_info[model_type]["decayPer"] and epoch % self.models_info[model_type]["decayPer"] == 0: adjust_learning_rate(self.models_info[model_type]["optimizer"], decayBy = self.models_info[model_type]["decayBy"])
			
	def train_models(self,
			model_list,
			pretrained_models = False,
			dataset_type = "HAM10000",
			lr_list = 1e-3,
			epochs = 200,
			batch_size = 64,
			starting_epoch = 1,
			version = 1,
			init_weights = True,
			decayPer = 25,
			decayBy = 0.75,
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
		self.starting_epoch = starting_epoch
		self.version = version
		self.init_weights = init_weights
		self.debug = debug
		self.total_epochs = epochs + starting_epoch - 1
		
		if not isinstance(pretrained_models, list): pretrained_models = [pretrained_models]*len(model_list)
		if not isinstance(lr_list, list): lr_list = [lr_list]*len(model_list)
		if not isinstance(decayPer, list): decayPer = [decayPer]*len(model_list)
		if not isinstance(decayBy, list): decayBy = [decayBy]*len(model_list)
			
		for item in [lr_list, model_list, pretrained_models]:
			assert len(item) == len(model_list), "Parameters mismatch"
		
		# Prepare dataset
		ds = selectData(dataset_type, batch_size = batch_size)
		train_dataset, train_dataloader, _,  val_dataloader = ds.getTrainData()
		if debug: print("Data Loaded")
			
		# setup models
		self.models_info = dict()
		for model_type, pretrained_model, lr, dB, dP in zip(model_list, pretrained_models, lr_list, decayBy, decayPer):
			model_name, save_folder, model, optimizer, loss_function, isContractive, isVariational = self.setup_model(model_type, dataset_type, lr, pretrained_model)
			self.models_info[model_type] = dict()
			self.models_info[model_type]["stop"] = False
			self.models_info[model_type]["model_name"] = model_name
			self.models_info[model_type]["save_path"] = save_folder
			self.models_info[model_type]["model"] = model
			if pretrained_model: load_model(self.models_info[model_type]["model"], pretrained_model)
			self.models_info[model_type]["optimizer"] = optimizer
			self.models_info[model_type]["loss_function"] = loss_function
			self.models_info[model_type]["train_loss"] = list()
			self.models_info[model_type]["val_loss"] = list()
			self.models_info[model_type]["decayBy"] = dB
			self.models_info[model_type]["decayPer"] = dP
			self.models_info[model_type]["step_function"] = self.step_normal
			if isContractive: self.models_info[model_type]["step_function"] = self.step_contractive
			if isVariational: self.models_info[model_type]["step_function"] = self.step_variational
			
			self.models_info[model_type]["history"] = {"train_loss" : list(), "val_loss": list()}
		
		# Start training
		for epoch in range(starting_epoch, self.total_epochs):
			batch_st = time()
			self.epoch_reset()
			
			# Train
			for batch_idx, batch_train_data in enumerate(train_dataloader, 0):
				# move data to device
				train_images, train_labels = batch_train_data
				train_images = train_images.to(Config.device)
				
				# forward pass: compute predictions
				for model_type in self.models_info.keys():
					if self.models_info[model_type]["stop"]: continue
					
					train_loss = self.models_info[model_type]["step_function"](self.models_info[model_type]["model"], self.models_info[model_type]["optimizer"], self.models_info[model_type]["loss_function"], train_images, isTrain = True)
					self.models_info[model_type]["train_loss"].append(train_loss)
				
			# validate
			for val_batch_idx, batch_val_data in enumerate(val_dataloader):
				with torch.no_grad():
					val_images, val_labels = batch_val_data
					val_images = val_images.to(Config.device)
					
					for model_type in self.models_info.keys():
						if self.models_info[model_type]["stop"]: continue
						val_loss = self.models_info[model_type]["step_function"](self.models_info[model_type]["model"], self.models_info[model_type]["optimizer"], self.models_info[model_type]["loss_function"], val_images, isTrain = False)
						self.models_info[model_type]["val_loss"].append(val_loss)
			
			for model_type in self.models_info.keys():
				# Record to history
				self.models_info[model_type]["history"]["train_loss"].append(np.mean(self.models_info[model_type]["train_loss"]))
				self.models_info[model_type]["history"]["val_loss"].append(np.mean(self.models_info[model_type]["val_loss"]))
						
			# Learning Rate Decay
			if np.sum(decayPer) > 1: self.lr_decay(epoch)
			# Print Status of the Epoch
			self.epoch_status(epoch, batch_st, time())
			# Save the latest models
			self.save_models(dataset_type, epoch == (self.total_epochs - 1))
			
		model_paths = [value["save_path"] for value in self.models_info.values()]
		
		try:
			del self.models_info, ds
			torch.cuda.empty_cache()
			gc.collect()
		except Exception as e:
			print("Could not clear the memory. Kill the process manually.")
			print(e)
			
		return model_paths