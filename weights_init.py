import torch
from torch import nn

def weights_init(m, initializer_type = "kaiming_uniform"):
	'''
	Initializes the weigts of the network
	'''
	
	initializer_type = initializer_type.lower()
	
	if "kaiming" in initializer_type:
		if "uniform" in initializer_type:
			init_fn = nn.init.kaiming_uniform_
		elif "normal" in initializer_type:
			init_fn = nn.init.kaiming_normal_
			
	if "xavier"	in initializer_type:
		if "uniform" in initializer_type:
			init_fn = nn.init.xavier_uniform_
		elif "normal" in initializer_type:
			init_fn = nn.init.xavier_normal_
			
	if isinstance(m, nn.Conv2d):
		init_fn(m.weight.data)
		try: m.bias.data.fill_(0.01)
		except: pass
	if isinstance(m, nn.Linear):
		m.weight.data.uniform_(0.0, 1.0)
		m.bias.data.fill_(0)
	if isinstance(m, nn.BatchNorm2d):
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)