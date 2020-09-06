import torch
from config import Config
from torch import nn

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim

	
class MSE_LOSS:
	def __init__(self, reduction = "sum"):
		# print("MSE")
		self.loss = nn.MSELoss(reduction = reduction)
	
	def __call__(self, original, reconstructions):
		return self.loss(original, reconstructions)

class BCE_LOSS:
	def __init__(self, reduction = "sum"):
		# print("BCE")
		self.loss = nn.BCELoss(reduction = reduction)
	
	def __call__(self, original, reconstructions):
		return self.loss(reconstructions, original)

class CONTRACTIVE_LOSS:
	def __init__(self, primary_loss = "mse", lamda = 1e-3):
		# print("CONT +", primary_loss)
		self.main_loss = MSE_LOSS(reduction = "mean")
		if "bce" in primary_loss: self.main_loss = BCE_LOSS(reduction = "mean")
		self.lamda = lamda
	
	def __call__(self, original, reconstructions, encodings, W):
		main_loss = self.main_loss(original, reconstructions)
		W = W.view(Config.EmbeddingSize, -1)
		dh = (encodings * (1 - encodings)).view(-1, Config.EmbeddingSize).to(Config.device)
		contractive_loss = self.lamda * torch.sum(dh**2 * torch.sum(W**2, dim=-1)).to(Config.device)
		return main_loss + contractive_loss

class VARIATIONAL_LOSS:
	def __init__(self):
		# print("VAR")
		self.bce_loss = BCE_LOSS(reduction = "mean")
		
	def __call__(self, original, reconstructions, mu, logvar):
		BCE = self.bce_loss(original, reconstructions)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return BCE + KLD, BCE, KLD

class MS_SSIM_LOSS(MS_SSIM):
	def forward(self, img1, img2):
		return 100*( 1 - super(MS_SSIM_LOSS, self).forward(img1, img2))

class SSIM_LOSS(SSIM):
	def forward(self, img1, img2):
		return 100*( 1 - super(SSIM_LOSS, self).forward(img1, img2))
	
class WEIGHTED_SIMILARITY:
	def __init__(self, primary_loss = "mse", weights = [1.0, 1.0]):
		self.main_loss = MSE_LOSS(reduction = "mean")
		if "bce" in primary_loss: self.main_loss = BCE_LOSS(reduction = "mean")
		self.ssim_loss = SSIM()
		self.weights = weights
	
	def __call__(self, original, reconstructions):
		return (self.weights[0] * self.main_loss(original, reconstructions)) + (self.weights[1] * 100 * (1 - self.ssim_loss.forward(original, reconstructions)))
			
	