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
# 		W = torch.flatten(W, start_dim = 1, end_dim = -1)
# 		dh = (encodings * (1 - encodings)).to(Config.device)
# 		dh = torch.squeeze(torch.squeeze(dh, dim = -1), dim = -1)
# 		contractive_loss = self.lamda * torch.sum(dh**2 * torch.sum(W**2, dim=-1)).to(Config.device)
		contractive_loss = (self.lamda * torch.norm(encodings*(1-encodings)) * torch.norm(W).to(Config.device))
		return main_loss + contractive_loss, {"MSE": main_loss, "CE": contractive_loss}

class VARIATIONAL_LOSS:
	def __init__(self):
		# print("VAR")
		self.bce_loss = BCE_LOSS(reduction = "mean")
		
	def __call__(self, original, reconstructions, mu, logvar):
		BCE = self.bce_loss(original, reconstructions)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return BCE + KLD, {"BCE": BCE, "KLD": KLD}
	
class WEIGHTED_SIMILARITY:
	def __init__(self, primary_loss = "mse", weights = [1.0, 1.0]):
		self.main_loss = MSE_LOSS(reduction = "mean")
		if "bce" in primary_loss: self.main_loss = BCE_LOSS(reduction = "mean")
		self.ssim_loss = SSIM()
		self.weights = weights
	
	def __call__(self, original, reconstructions):
		return (self.weights[0] * self.main_loss(original, reconstructions)) + (self.weights[1] * 100 * (1 - self.ssim_loss.forward(original, reconstructions)))
			
class MAHALANOBIS_LOSS(nn.Module):
	def __init__(self, reduction = "mean"):
		super(MAHALANOBIS_LOSS, self).__init__()
		self.reduction = "mean"

	def getCovariance(self, x):
		return (1 / (x.size(0) - 1)) * x.t().mm(x)
    
	def calcMahalanobis(self, X):
		X -= torch.mean(X, dim=0)
		s_inv = torch.pinverse(self.getCovariance(X))
		md = torch.diag(torch.mm(torch.mm(X, s_inv), X.t()))
		return md

	def forward(self, X):
		md = self.calcMahalanobis(X)
		if "mean" in self.reduction: return torch.mean(md)
		elif "sum" in self.reduction: return torch.sum(md)

class MahalanobisLayer(nn.Module):
    def __init__(self, dim = 300):
        super(MahalanobisLayer, self).__init__()
        self.register_buffer('S', torch.eye(dim))
        self.register_buffer('S_inv', torch.eye(dim))

    def cov(self, x):
        x -= torch.mean(x, dim=0)
        return 1 / (x.size(0) - 1) * x.t().mm(x)

    def forward(self, x):
        delta = x - torch.mean(x, dim = 0)
        self.S_inv = torch.pinverse(self.cov(delta + 1e-10))
        m = torch.mm(torch.mm(delta, self.S_inv), delta.t())
        return torch.mean(torch.diag(m))

class MANIFOLD_LOSS:
	def __init__(self, primary_loss = "mse", weights = [1.0, 1.0]):
		self.main_loss = MSE_LOSS(reduction = "mean")
		if "bce" in primary_loss: self.main_loss = BCE_LOSS(reduction = "mean")
		self.mahalanobis_loss = MahalanobisLayer()
		self.weights = weights
	
	def __call__(self, original, reconstruction, encoding):
		return (self.weights[0] * self.main_loss(original, reconstruction)) + (self.weights[1] * self.mahalanobis_loss(encoding))