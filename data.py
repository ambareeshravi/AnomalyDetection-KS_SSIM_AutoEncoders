import torch
from glob import glob

import os, cv2, shutil
import pickle as pkl, json
import numpy as np, pandas as pd

import torchvision.datasets as dset
import torchvision.transforms as transforms

from config import Config

class HAM10000_Dataset:
	def __init__(self, data_path = "../HAM10000_SPLIT/", batch_size = 32):
		self.data_path = data_path
		self.batch_size = batch_size
		self.train_data_transform = transforms.Compose([        
					transforms.RandomGrayscale(p = 0.25),
					transforms.RandomHorizontalFlip(p=0.25),
					transforms.RandomRotation(10),
					transforms.Resize(Config.ImageSize),
					transforms.CenterCrop(Config.ImageSize),
					transforms.ToTensor(),
# 					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
						])
	
		self.test_data_transform = transforms.Compose([
					transforms.Resize(Config.ImageSize),
					transforms.CenterCrop(Config.ImageSize),
					transforms.ToTensor(),
# 					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
							])

	def getTrainData(self, val_split = 0.9):
		train_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "NORMAL"), transform = self.train_data_transform)
		self.classes = train_dataset.classes
		
		split_point = int(len(train_dataset)*val_split)
		train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [split_point, len(train_dataset) - split_point])
		
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

		return train_dataset, train_dataloader, None, val_dataloader

	def getTestData(self, test_batch_size = 100):
		normal_test_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "NORMAL_TEST"), transform = self.test_data_transform)
		normal_test_dataloader = torch.utils.data.DataLoader(normal_test_dataset, batch_size = test_batch_size, shuffle=True, pin_memory=True)
		
		abnormal_dataloaders = dict()
		for folder in glob(os.path.join(self.data_path, "ABNORMAL/*")):
			category = os.path.split(folder)[-1]
			abnormal_test_dataset = dset.ImageFolder(root=folder, transform = self.test_data_transform)
			abnormal_test_dataloader = torch.utils.data.DataLoader(abnormal_test_dataset, batch_size = test_batch_size, shuffle = True, pin_memory=True)
			abnormal_dataloaders[category] = abnormal_test_dataloader
			
		return normal_test_dataset, normal_test_dataloader, abnormal_dataloaders
	
	
class DISTRACTION_Dataset:
	def __init__(self, data_path = "../../datasets/IR_DISTRACTION/", batch_size = 64):
		self.data_path = data_path
		self.batch_size = batch_size
		self.train_data_transform = transforms.Compose([        
					transforms.RandomGrayscale(p = 0.25),
					transforms.RandomHorizontalFlip(p=0.25),
					transforms.RandomRotation(10),
					transforms.Resize(Config.ImageSize),
					transforms.CenterCrop(Config.ImageSize),
					transforms.ToTensor(),
# 					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
						])
	
		self.test_data_transform = transforms.Compose([
					transforms.Resize(Config.ImageSize),
					transforms.CenterCrop(Config.ImageSize),
					transforms.ToTensor(),
# 					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
							])

	def getTrainData(self, val_split = 0.95):
		train_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "NORMAL_TRAIN"), transform = self.train_data_transform)
		self.classes = train_dataset.classes
		
		split_point = int(len(train_dataset)*val_split)
		train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [split_point, len(train_dataset) - split_point])
		
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

		return train_dataset, train_dataloader, None, val_dataloader

	def getTestData(self, test_batch_size = 100):
		normal_test_dataset = dset.ImageFolder(root=os.path.join(self.data_path, "NORMAL_TEST"), transform = self.test_data_transform)
		normal_test_dataloader = torch.utils.data.DataLoader(normal_test_dataset, batch_size = test_batch_size, shuffle=True, pin_memory=True)
		
		abnormal_dataloaders = dict()
		for folder in glob(os.path.join(self.data_path, "ABNORMAL/*")):
			category = os.path.split(folder)[-1]
			abnormal_test_dataset = dset.ImageFolder(root=folder, transform = self.test_data_transform)
			abnormal_test_dataloader = torch.utils.data.DataLoader(abnormal_test_dataset, batch_size = test_batch_size, shuffle = True, pin_memory=True)
			abnormal_dataloaders[category] = abnormal_test_dataloader
			
		return normal_test_dataset, normal_test_dataloader, abnormal_dataloaders