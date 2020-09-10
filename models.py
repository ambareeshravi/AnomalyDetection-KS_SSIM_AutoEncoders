import numpy as np
import torch
from torch import nn
from torchvision import models
from config import Config

np.random.seed(0)
torch.manual_seed(0)

def get_Activation(activation_type = "relu", negative_slope = 0.2):
	if "sigmoid" in activation_type.lower(): return nn.Sigmoid()
	elif "leaky" in activation_type.lower(): return nn.LeakyReLU(negative_slope = negative_slope, inplace = True)
	elif "relu" in activation_type.lower(): return nn.ReLU(inplace = True)
	elif "elu" in activation_type.lower(): return nn.ELU(inplace = True)
	elif "tanh" in activation_type.lower(): return nn.Tanh()
    
def get_ConvBlock(in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, useBatchNorm = True, activation_type = "relu"):
	layers = [
		nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
	]
	if useBatchNorm: layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
	if activation_type: layers.append(get_Activation(activation_type = activation_type))
	return nn.Sequential(*layers)
    
def get_ConvTransposeBlock(in_channels, out_channels, kernel_size, stride = 1, padding = 0, output_padding = 0, dilation = 1, useBatchNorm = True, activation_type = "relu"):
	layers = [
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding = output_padding, dilation = dilation)
	]
	if useBatchNorm: layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
	if activation_type: layers.append(get_Activation(activation_type = activation_type))
	return nn.Sequential(*layers)

# Qiang's AutoEncoder
class Qiang_AutoEncoder(nn.Module):
	'''
	AutoEncoder Implementation by Qiang et. al based on "Anomaly Detection for Images using Auto-Encoder based Sparse Representation"
	Author's code
	'''
	def __init__(self):
		super(Qiang_AutoEncoder, self).__init__()
		# print(self)
		self.encoder = nn.Sequential(
			nn.Conv2d(Config.channels, Config.ImageSize, 4, 4, 0, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(Config.ImageSize, Config.ImageSize * 4, 4, 4, 0, bias=False),
			nn.BatchNorm2d(Config.ImageSize * 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(Config.ImageSize * 4, Config.ImageSize * 16, 4, 4, 0, bias=False),
			nn.BatchNorm2d(Config.ImageSize * 16),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(Config.ImageSize * 16, Config.EmbeddingSize, 2, 1, 0, bias=False),
			nn.Sigmoid()
		)  
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(Config.EmbeddingSize, Config.ImageSize * 16, 2, 1, 0, bias=False),
			nn.BatchNorm2d(Config.ImageSize * 16),
			nn.ReLU(True),
			nn.ConvTranspose2d(Config.ImageSize * 16, Config.ImageSize * 4, 4, 4, 0, bias=False),
			nn.BatchNorm2d(Config.ImageSize * 4),
			nn.ReLU(True),
			nn.ConvTranspose2d(Config.ImageSize * 4, Config.ImageSize, 4, 4, 0, bias=False),
			nn.BatchNorm2d(Config.ImageSize),
			nn.ReLU(True),
			nn.ConvTranspose2d(Config.ImageSize, Config.channels, 4, 4, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, inp):
		encoding = self.encoder(inp)
		reconstruction = self.decoder(encoding)
		return reconstruction, encoding

# Base AutoEncoder
class AE_v0(nn.Module):
	'''
	Modified, simple AutoEncoder
	'''
	def __init__(self, full = False):
		super(AE_v0, self).__init__()
		# print(self)
		self.filter_count = [64, 128, 256, 512]

		self.encoder = nn.Sequential(
			get_ConvBlock(in_channels = Config.channels, out_channels = self.filter_count[0], kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False),
			get_ConvBlock(in_channels = self.filter_count[0], out_channels = self.filter_count[1], kernel_size = 3, stride = 2, padding =  0, activation_type = "leaky_relu"),
			get_ConvBlock(in_channels = self.filter_count[1], out_channels = self.filter_count[2], kernel_size = 5, stride = 3, padding =  0, activation_type = "leaky_relu"),
			get_ConvBlock(in_channels = self.filter_count[2], out_channels = self.filter_count[3], kernel_size = 5, stride = 2, padding =  0, activation_type = "leaky_relu"),
			get_ConvBlock(in_channels = self.filter_count[3], out_channels = Config.EmbeddingSize, kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False, activation_type = "sigmoid"),
			)

		self.decoder = nn.Sequential(
			get_ConvTransposeBlock(in_channels = Config.EmbeddingSize, out_channels = self.filter_count[3], kernel_size = 3, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[3], out_channels = self.filter_count[2], kernel_size = 5, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[2], out_channels = self.filter_count[1], kernel_size = 5, stride = 3, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[1], out_channels = self.filter_count[0], kernel_size = 3, stride = 2, padding = 0, dilation = 2),
			get_ConvTransposeBlock(in_channels = self.filter_count[0], out_channels = Config.channels, kernel_size = 3, stride = 2, padding = 0, dilation = 3, output_padding= 1, useBatchNorm = False, activation_type = "sigmoid")
		)

	def forward(self, inp):
		encoding = self.encoder(inp)
		reconstruction = self.decoder(encoding)
		return reconstruction, encoding
	
class AE_v1(nn.Module):
	'''
	Modified and improved simple AutoEncoder - version 1
	'''
	def __init__(self):
		super(AE_v1, self).__init__()
		# print(self)
		self.filter_count = [64, 128, 256, 512]

		self.encoder = nn.Sequential(
			get_ConvBlock(in_channels = Config.channels, out_channels = self.filter_count[0], kernel_size = 2, stride = 2, padding =  0, useBatchNorm = False),
			get_ConvBlock(in_channels = self.filter_count[0], out_channels = self.filter_count[1], kernel_size = 2, stride = 2, padding =  0),
			get_ConvBlock(in_channels = self.filter_count[1], out_channels = self.filter_count[2], kernel_size = 4, stride = 4, padding =  0),
			get_ConvBlock(in_channels = self.filter_count[2], out_channels = self.filter_count[3], kernel_size = 4, stride = 4, padding =  0),
			get_ConvBlock(in_channels = self.filter_count[3], out_channels = Config.EmbeddingSize, kernel_size = 2, stride = 1, padding =  0),
			)

		self.decoder = nn.Sequential(
			get_ConvTransposeBlock(in_channels = Config.EmbeddingSize, out_channels = self.filter_count[3], kernel_size = 2, stride = 1, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[3], out_channels = self.filter_count[2], kernel_size = 4, stride = 4, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[2], out_channels = self.filter_count[1], kernel_size = 4, stride = 4, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[1], out_channels = self.filter_count[0], kernel_size = 2, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[0], out_channels = Config.channels, kernel_size = 2, stride = 2, padding = 0, activation_type = "sigmoid"),
		)

	def forward(self, inp):
		encoding = self.encoder(inp)
		reconstruction = self.decoder(encoding)
		return reconstruction, encoding

class AE(nn.Module):
	'''
	Modified, simple AutoEncoder
	'''
	def __init__(self, full = False):
		super(AE, self).__init__()
		# print(self)
		self.filter_count = [64, 128, 256, 512]

		self.encoder = nn.Sequential(
			get_ConvBlock(in_channels = Config.channels, out_channels = self.filter_count[0], kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False),
			get_ConvBlock(in_channels = self.filter_count[0], out_channels = self.filter_count[1], kernel_size = 3, stride = 2, padding =  0, activation_type = "leaky_relu"),
			get_ConvBlock(in_channels = self.filter_count[1], out_channels = self.filter_count[2], kernel_size = 5, stride = 3, padding =  0, activation_type = "leaky_relu"),
			get_ConvBlock(in_channels = self.filter_count[2], out_channels = self.filter_count[3], kernel_size = 5, stride = 2, padding =  0, activation_type = "leaky_relu"),
			get_ConvBlock(in_channels = self.filter_count[3], out_channels = Config.EmbeddingSize, kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False, activation_type = "sigmoid"),
			)

		self.decoder = nn.Sequential(
			get_ConvTransposeBlock(in_channels = Config.EmbeddingSize, out_channels = self.filter_count[3], kernel_size = 3, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[3], out_channels = self.filter_count[2], kernel_size = 5, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[2], out_channels = self.filter_count[1], kernel_size = 5, stride = 3, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[1], out_channels = self.filter_count[0], kernel_size = 5, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[0], out_channels = self.filter_count[0]//4, kernel_size = 5, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[0] // 4, out_channels = Config.channels, kernel_size = 4, stride = 1, padding = 0, useBatchNorm = False, activation_type = "sigmoid")
		)

	def forward(self, inp):
		encoding = self.encoder(inp)
		reconstruction = self.decoder(encoding)
		return reconstruction, encoding

# Contractive
# Derive Base Encoder and use the same

# ACB
class ACB(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0 , dilation = 1, useBatchNorm = True, activation_type = "relu"):
		super(ACB, self).__init__()
		s_layers = [
			nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
					  stride = stride, padding = (kernel_size//2, kernel_size//2), dilation = dilation)
		]
		if useBatchNorm: s_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.s = nn.Sequential(*s_layers)

		h_layers = [
			nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (1, kernel_size),
					  stride = stride, padding = (0, kernel_size//2), dilation = dilation)
		]
		if useBatchNorm: h_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.h = nn.Sequential(*h_layers)

		v_layers = [
			nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (kernel_size, 1),
					  stride = stride, padding = (kernel_size // 2, 0), dilation = dilation)
		]
		if useBatchNorm: v_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.v = nn.Sequential(*v_layers)

		if activation_type: self.activation = get_Activation(activation_type)
		else: self.activation = lambda x: x

	def forward(self, x):
		return self.activation(self.s(x) + self.h(x) + self.v(x))

class ADB(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, output_padding = 0,  useBatchNorm = True, activation_type = "relu"):
		super(ADB, self).__init__()
		s_layers = [
			nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, output_padding = output_padding)
		]
		if useBatchNorm: s_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.s = nn.Sequential(*s_layers)

		h_layers = [
			nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (1, kernel_size), stride = stride, padding = padding, dilation = (kernel_size, 1), output_padding = (kernel_size-1, 0))
		]
		if useBatchNorm: h_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.h = nn.Sequential(*h_layers)

		v_layers = [
			nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (kernel_size, 1), stride = stride, padding = padding, dilation = (1, kernel_size), output_padding = (0, kernel_size-1))
		]
		if useBatchNorm: v_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.v = nn.Sequential(*v_layers)

		if activation_type: self.activation = get_Activation(activation_type)
		else: self.activation = lambda x: x

	def forward(self, x):
		return self.activation(self.s(x) + self.h(x) + self.v(x))

class KS_AE_v0(AE):
	def __init__(self, normalDecoder = False):
		super(KS_AE_v0, self).__init__()
		# print(self)
		self.filter_count = [64, 64, 128, 256]

		self.encoder = nn.Sequential(
			ACB(in_channels = Config.channels, out_channels = self.filter_count[0], kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False),
			ACB(in_channels = self.filter_count[0], out_channels = self.filter_count[1], kernel_size = 3, stride = 2, padding =  0),
			ACB(in_channels = self.filter_count[1], out_channels = self.filter_count[2], kernel_size = 5, stride = 3, padding =  0),
			ACB(in_channels = self.filter_count[2], out_channels = self.filter_count[3], kernel_size = 5, stride = 3, padding =  0),
			get_ConvBlock(in_channels = self.filter_count[3], out_channels = Config.EmbeddingSize, kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False, activation_type = "sigmoid"),
			)
		
		if not normalDecoder:
			self.decoder = nn.Sequential(
				ADB(in_channels = Config.EmbeddingSize, out_channels = self.filter_count[3], kernel_size = 3, stride = 2, padding = 0),
				ADB(in_channels = self.filter_count[3], out_channels = self.filter_count[2], kernel_size = 5, stride = 2, padding = 0),
				ADB(in_channels = self.filter_count[2], out_channels = self.filter_count[1], kernel_size = 5, stride = 3, padding = 0),
				ADB(in_channels = self.filter_count[1], out_channels = self.filter_count[0], kernel_size = 3, stride = 2, padding = 0),
				get_ConvTransposeBlock(in_channels = self.filter_count[0], out_channels = Config.channels, kernel_size = 3, stride = 2, padding = 0, dilation = 5, output_padding= 1, useBatchNorm = False, activation_type = "sigmoid")
			)

	def forward(self, x):
		encoding = self.encoder(x)
		reconstruction = self.decoder(encoding)
		return reconstruction, encoding

class KS_AE(AE):
	def __init__(self, normalDecoder = False):
		super(KS_AE, self).__init__()
		# print(self)
		self.filter_count = [64, 64, 128, 256]

		self.encoder = nn.Sequential(
			ACB(in_channels = Config.channels, out_channels = self.filter_count[0], kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False),
			ACB(in_channels = self.filter_count[0], out_channels = self.filter_count[1], kernel_size = 3, stride = 2, padding =  0),
			ACB(in_channels = self.filter_count[1], out_channels = self.filter_count[2], kernel_size = 5, stride = 3, padding =  0),
			ACB(in_channels = self.filter_count[2], out_channels = self.filter_count[3], kernel_size = 5, stride = 3, padding =  0),
			get_ConvBlock(in_channels = self.filter_count[3], out_channels = Config.EmbeddingSize, kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False, activation_type = "sigmoid"),
			)
		
		if not normalDecoder:
			self.decoder = nn.Sequential(
				ADB(in_channels = Config.EmbeddingSize, out_channels = self.filter_count[3], kernel_size = 3, stride = 2, padding = 0),
				ADB(in_channels = self.filter_count[3], out_channels = self.filter_count[2], kernel_size = 5, stride = 2, padding = 0),
				ADB(in_channels = self.filter_count[2], out_channels = self.filter_count[1], kernel_size = 5, stride = 3, padding = 0),
				ADB(in_channels = self.filter_count[1], out_channels = self.filter_count[0], kernel_size = 5, stride = 2, padding = 0),
				ADB(in_channels = self.filter_count[0], out_channels = self.filter_count[0]//4, kernel_size = 5, stride = 2, padding = 0),
				get_ConvTransposeBlock(in_channels = self.filter_count[0]//4, out_channels = Config.channels, kernel_size = 4, stride = 1, padding = 0, useBatchNorm = False, activation_type = "sigmoid")
			)

	def forward(self, x):
		encoding = self.encoder(x)
		reconstruction = self.decoder(encoding)
		return reconstruction, encoding

# Variatonal
class VAE(AE):
	def __init__(self):
		super(VAE, self).__init__()
		# print(self)
		self.parameter_size = Config.EmbeddingSize
		self.mean_fc = nn.Linear(Config.EmbeddingSize, self.parameter_size)
		self.var_fc = nn.Linear(Config.EmbeddingSize, self.parameter_size)
	
	def encode(self, x):
		e = self.encoder(x).view(-1, Config.EmbeddingSize)
		mean = nn.ReLU()(self.mean_fc(e))
		var = nn.ReLU()(self.var_fc(e))
		return mean, var
	
	def decode(self, encoding):
		return self.decoder(encoding.view(-1, Config.EmbeddingSize, 1, 1))
	
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

# Variatonal
class KS_VAE(KS_AE):
	def __init__(self):
		super(KS_VAE, self).__init__()
		# print(self)
		self.parameter_size = Config.EmbeddingSize
		self.mean_fc = nn.Linear(Config.EmbeddingSize, self.parameter_size)
		self.var_fc = nn.Linear(Config.EmbeddingSize, self.parameter_size)
	
	def encode(self, x):
		e = self.encoder(x).view(-1, Config.EmbeddingSize)
		mean = nn.ReLU()(self.mean_fc(e))
		var = nn.ReLU()(self.var_fc(e))
		return mean, var
	
	def decode(self, encoding):
		return self.decoder(encoding.view(-1, Config.EmbeddingSize, 1, 1))
	
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

# Split
def BN_R(n, p = 0.2):
	'''
	Activation Block: ReLU(BN(x))

	Args:
		n - number of output channels as <int>
		p - probability of dropout as <float>
	Returns:
		<nn.Sequential> block
	'''
	return nn.Sequential(
		nn.BatchNorm2d(n), #, momentum = None, track_running_stats = False),
		nn.ReLU6(inplace = True), # Found ReLU6 to be performing better thatn ReLU (Improvement)
		nn.Dropout2d(p) # Slows down training but better generalization (Improvement)
	)

class SM_AE(AE):
	def __init__(self, normalDecoder = False, ngpu = 1):
		super(SM_AE, self).__init__()
		self.ngpu = ngpu

		self.filter_count = [64, 128, 256, 512]

		# Encoder Layers
		# SM Block 1
		self.lconv1 = nn.Conv2d(Config.channels, self.filter_count[0], 3, 2)
		self.rconv1 = nn.Conv2d(Config.channels, self.filter_count[0], 4, 2)
		self.act_block1 = nn.ReLU(self.filter_count[0])

		# SM Block 2
		self.lconv2 = nn.Conv2d(self.filter_count[0], self.filter_count[1], 3, 2)
		self.rconv2 = nn.Conv2d(self.filter_count[0], self.filter_count[1], 4, 2, padding_mode='replicate', padding=1)
		self.act_block2 = BN_R(self.filter_count[1])

		# SM Block 3
		self.lconv3 = nn.Conv2d(self.filter_count[1], self.filter_count[2], 4, 3)
		self.rconv3 = nn.Conv2d(self.filter_count[1], self.filter_count[2], 5, 3, padding_mode='replicate', padding=1)
		self.act_block3 = BN_R(self.filter_count[2])

		# SM Block 4
		self.lconv4 = nn.Conv2d(self.filter_count[2], self.filter_count[3], 4, 3)
		self.rconv4 = nn.Conv2d(self.filter_count[2], self.filter_count[3], 5, 3, padding_mode='replicate', padding=1)
		self.act_block4 = BN_R(self.filter_count[3])

		# SM Block 5
		self.lconv5 = nn.Conv2d(self.filter_count[3], Config.EmbeddingSize, 3, 2)
		self.rconv5 = nn.Conv2d(self.filter_count[3], Config.EmbeddingSize, 4, 2, padding_mode='replicate', padding=1)	
		self.act_block5 = nn.Sigmoid()

		# Decoder layers
		# SM Block 6
		self.ltconv1 = nn.ConvTranspose2d(Config.EmbeddingSize, self.filter_count[-1], 3, 2, output_padding = 1)
		self.rtconv1 = nn.ConvTranspose2d(Config.EmbeddingSize, self.filter_count[-1], 4, 2)
		self.act_block6 = BN_R(self.filter_count[-1])

		# SM Block 7
		self.ltconv2 = nn.ConvTranspose2d(self.filter_count[-1], self.filter_count[-2], 4, 3, output_padding = 1)
		self.rtconv2 = nn.ConvTranspose2d(self.filter_count[-1], self.filter_count[-2], 5, 3)
		self.act_block7 = BN_R(self.filter_count[-2])

		# SM Block 8
		self.ltconv3 = nn.ConvTranspose2d(self.filter_count[-2], self.filter_count[-3], 4, 2, output_padding = 1)
		self.rtconv3 = nn.ConvTranspose2d(self.filter_count[-2], self.filter_count[-3], 5, 2)
		self.act_block8 = BN_R(self.filter_count[-3])

		# SM Block 9
		self.ltconv4 = nn.ConvTranspose2d(self.filter_count[-3], self.filter_count[-4], 3, 2, output_padding = 1)
		self.rtconv4 = nn.ConvTranspose2d(self.filter_count[-3], self.filter_count[-4], 4, 2)
		self.act_block9 = BN_R(self.filter_count[-4])

		# SM Block 9
		self.ltconv5 = nn.ConvTranspose2d(self.filter_count[-4], Config.channels, 4, 2, padding = 1)
		self.act_block10 = nn.Sequential(
			nn.BatchNorm2d(Config.channels),
			nn.Sigmoid()
		)

		if not normalDecoder: self.decoder_fn = self.sm_decoder
		else: self.decoder_fn = self.decoder

	def encoder(self, x):
		l1_a = self.act_block1((self.lconv1(x) + self.rconv1(x)))
		l2_a = self.act_block2((self.lconv2(l1_a) + self.rconv2(l1_a)))
		l3_a = self.act_block3((self.lconv3(l2_a) + self.rconv3(l2_a)))
		l4_a = self.act_block4((self.lconv4(l3_a) + self.rconv4(l3_a)))
		l5_a = self.act_block5((self.lconv5(l4_a) + self.rconv5(l4_a)))
		return l5_a

	def sm_decoder(self, x):
		l6_a = self.act_block6(self.ltconv1(x) + self.rtconv1(x))
		l7_a = self.act_block7(self.ltconv2(l6_a) + self.rtconv2(l6_a))
		l8_a = self.act_block8(self.ltconv3(l7_a) + self.rtconv3(l7_a))
		l9_a = self.act_block9(self.ltconv4(l8_a) + self.rtconv4(l8_a))
		l10_a = self.act_block10(self.ltconv5(l9_a))
		return l10_a

	def forward(self, x):
		encoding = self.encoder(x)
		reconstruction = self.decoder_fn(encoding)
		return reconstruction, encoding
	
	
class SM_AE_V2(AE):
	def __init__(self, normalDecoder = False, ngpu = 1):
		super(SM_AE_V2, self).__init__()
		self.ngpu = ngpu

		self.filter_count = [32, 64, 128, 256] #, 512]

		# Encoder Layers
		# SM Block 1
		self.lconv1 = nn.Conv2d(Config.channels, self.filter_count[0], 3, 2)
		self.rconv1 = nn.Conv2d(Config.channels, self.filter_count[0], 4, 2)
		self.act_block1 = nn.ReLU(self.filter_count[0])

		# SM Block 2
		self.lconv2 = nn.Conv2d(self.filter_count[0], self.filter_count[1], 3, 2)
		self.rconv2 = nn.Conv2d(self.filter_count[0], self.filter_count[1], 4, 2, padding_mode='replicate', padding=1)
		self.act_block2 = BN_R(self.filter_count[1])

		# SM Block 3
		self.lconv3 = nn.Conv2d(self.filter_count[1], self.filter_count[2], 4, 3)
		self.rconv3 = nn.Conv2d(self.filter_count[1], self.filter_count[2], 5, 3, padding_mode='replicate', padding=1)
		self.act_block3 = BN_R(self.filter_count[2])

		# SM Block 4
		self.lconv4 = nn.Conv2d(self.filter_count[2], self.filter_count[3], 4, 3)
		self.rconv4 = nn.Conv2d(self.filter_count[2], self.filter_count[3], 5, 3, padding_mode='replicate', padding=1)
		self.act_block4 = BN_R(self.filter_count[3])

		# SM Block 5
		self.lconv5 = nn.Conv2d(self.filter_count[3], Config.EmbeddingSize, 3, 2)
		self.rconv5 = nn.Conv2d(self.filter_count[3], Config.EmbeddingSize, 4, 2, padding_mode='replicate', padding=1)	
		self.act_block5 = nn.Sigmoid()

		# Decoder layers
		# SM Block 6
		self.ltconv1 = nn.ConvTranspose2d(Config.EmbeddingSize, self.filter_count[-1], 3, 2, output_padding = 1)
		self.rtconv1 = nn.ConvTranspose2d(Config.EmbeddingSize, self.filter_count[-1], 4, 2)
		self.act_block6 = BN_R(self.filter_count[-1])

		# SM Block 7
		self.ltconv2 = nn.ConvTranspose2d(self.filter_count[-1], self.filter_count[-2], 4, 3, output_padding = 1)
		self.rtconv2 = nn.ConvTranspose2d(self.filter_count[-1], self.filter_count[-2], 5, 3)
		self.act_block7 = BN_R(self.filter_count[-2])

		# SM Block 8
		self.ltconv3 = nn.ConvTranspose2d(self.filter_count[-2], self.filter_count[-3], 4, 2, output_padding = 1)
		self.rtconv3 = nn.ConvTranspose2d(self.filter_count[-2], self.filter_count[-3], 5, 2)
		self.act_block8 = BN_R(self.filter_count[-3])

		# SM Block 9
		self.ltconv4 = nn.ConvTranspose2d(self.filter_count[-3], self.filter_count[-4], 3, 2, output_padding = 1)
		self.rtconv4 = nn.ConvTranspose2d(self.filter_count[-3], self.filter_count[-4], 4, 2)
		self.act_block9 = BN_R(self.filter_count[-4])

		# SM Block 9
		self.ltconv5 = nn.ConvTranspose2d(self.filter_count[-4], Config.channels, 4, 2, padding = 1)
		self.act_block10 = nn.Sequential(
			nn.BatchNorm2d(Config.channels),
			nn.Sigmoid()
		)

		if not normalDecoder: self.decoder_fn = self.sm_decoder
		else: self.decoder_fn = self.decoder

	def encoder(self, x):
		l1_a = self.act_block1((self.lconv1(x) + self.rconv1(x)))
		l2_a = self.act_block2((self.lconv2(l1_a) + self.rconv2(l1_a)))
		l3_a = self.act_block3((self.lconv3(l2_a) + self.rconv3(l2_a)))
		l4_a = self.act_block4((self.lconv4(l3_a) + self.rconv4(l3_a)))
		l5_a = self.act_block5((self.lconv5(l4_a) + self.rconv5(l4_a)))
		return l5_a

	def sm_decoder(self, x):
		l6_a = self.act_block6(self.ltconv1(x) + self.rtconv1(x))
		l7_a = self.act_block7(self.ltconv2(l6_a) + self.rtconv2(l6_a))
		l8_a = self.act_block8(self.ltconv3(l7_a) + self.rtconv3(l7_a))
		l9_a = self.act_block9(self.ltconv4(l8_a) + self.rtconv4(l8_a))
		l10_a = self.act_block10(self.ltconv5(l9_a))
		return l10_a

	def forward(self, x):
		encoding = self.encoder(x)
		reconstruction = self.decoder_fn(encoding)
		return reconstruction, encoding