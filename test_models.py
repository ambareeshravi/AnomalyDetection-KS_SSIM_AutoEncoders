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

from torchvision.utils import save_image
import torchvision.utils as vutils

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

from sklearn.metrics import roc_auc_score

np.random.seed(0)
torch.manual_seed(0)

class Tester:
	def __init__(self, dataset_type = "HAM10000", batch_size = 100, n_images = 5):
		self.ds = selectData(dataset_type, batch_size = batch_size)
		self.normal_test_dataset, self.normal_test_dataloader, self.abnormal_dataloaders = self.ds.getTestData()
		self.n_images = n_images
	
	def save_reconstruction(self, image, recons, file_name = "", isNormal = True):
		vutils.save_image(recons.detach(), file_name + "_" + str(isNormal) + "_recons.jpg", normalize=True)
		vutils.save_image(image.detach(), file_name + "_" + str(isNormal) + "_original.jpg", normalize=True)

	def step(self, images, idx):
		images = images.to(Config.device)
		
		if self.isVariational:
			reconstruction, mu, logvar = self.test_model(images)
			loss, components = self.loss_function(images, reconstruction, mu, logvar)
		elif self.isContractive:
			reconstruction, encoding = self.test_model(images)
			loss, components = self.loss_function(images, reconstruction, encoding, self.test_model.encoder[-1][0].weight)
		else:
			reconstruction, encoding = self.test_model(images)
			loss = self.loss_function(images, reconstruction)
                
		if self.count in self.random_indices:
			self.save_reconstruction(images[-1], reconstruction[-1], file_name = os.path.join(self.model_dir, str(self.count))[:-1], isNormal = idx < 100)
		self.count += 1
			
		return loss.item()

	def test(self, model_dir, test_run = 10, save_random_recons = True):
		'''
		Tests the given model untrained normal images and random abnormal images
		'''
		self.count = 0
		self.random_indices = np.random.choice(range(500), self.n_images)
		st = time()
		model_path = sorted(glob(os.path.join(model_dir, "*.tar")))[-1]
		results_file = open(os.path.join(model_dir, os.path.split(model_dir)[-1] + "_results.txt"), "w+")

		self.isVariational, self.isContractive = False, False
		if "variational" in model_path: self.isVariational = True
		if "contractive" in model_path: self.isContractive = True
		self.test_model, self.loss_function = selectModel(model_path)
		load_model(self.test_model, model_path)
		self.test_model.to(Config.device)
		self.test_model.eval()
		
		self.model_dir = os.path.split(model_path)[0]

		test_results = list()
		
		for run in range(1, test_run + 1):
			total = list()
			results = dict()
			
			for idx, normal_data in enumerate(self.normal_test_dataloader):
				normal_images = normal_data[0]
				break
			
			abnormal_images = dict()
			for category, dataloader in self.abnormal_dataloaders.items():
				for idx, abnormal_data in enumerate(dataloader):
					abnormal_images[category] = abnormal_data[0]
					break
				
			for tp, test_images in abnormal_images.items():
				images = torch.Tensor(np.concatenate((normal_images.cpu().numpy(), test_images.cpu().numpy())))
				labels = np.array([0] * 100 + [1] * 100)
				with torch.no_grad():
					losses = [self.step(torch.unsqueeze(image, axis = 0), idx) for idx, image in enumerate(images)]
				
				AUCROC_score = roc_auc_score(labels, losses)
				total.append(AUCROC_score)
				results[tp] = AUCROC_score
			results_file.write("Average for RUN:"+str(run)+"->"+str(sum(total)/len(total))+"\n")
			test_results.append(results)

		results_df = pd.DataFrame(test_results)
		results_file.write("\nPer class average of 10 runs:\n" + str(results_df.mean(axis=0)) + "\n")
		results_file.write("\nOVERALL MEAN AUC ROC SCORE: "+str(np.mean(results_df.mean(axis=0))) + "\n")
		results_file.write("\n\n")
		results_file.write(str(results_df))
		results_file.close()
		print("Took ", (time()-st), "seconds for", model_dir)
		
		try:
			del self.test_model, self.loss_function
		except:
			pass