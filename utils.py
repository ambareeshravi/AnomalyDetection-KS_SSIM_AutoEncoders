import torch
from torch import nn
from sklearn.metrics import accuracy_score
from models import *
from data import *
from losses import *
from glob import glob

def save_model(model, model_name):
	'''
	Saves model object to a path
	'''
	if ".tar" not in model_name: model_name += ".tar"
	torch.save({'state_dict': model.state_dict()}, model_name)

def load_model(model, model_name):
	'''
	Load model parameters to the model object
	'''
	if ".tar" not in model_name: model_name += ".tar"
	checkpoint = torch.load(model_name, map_location = 'cpu')
	model.load_state_dict(checkpoint['state_dict'])

def adjust_learning_rate(optimizer, decayBy = 0.9):
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * decayBy
	print("Adjusted Learning Rate:", param_group['lr'])

# def calc_accuracy(labels, predictions, argmaxReq = False):
# 	'''
# 	Calculates the accuracy given labels and predictions
# 	'''
# 	labels = labels.detach().cpu().numpy()
# 	predictions = predictions.detach().cpu().numpy()
# 	if argmaxReq: predictions = np.argmax(predictions, axis = -1)
# 	return accuracy_score(labels.flatten(), np.round(predictions.flatten()))

def getPredictions(y_pred):
	y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
	_, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
	return y_pred_tags

def calc_accuracy(y_test, y_pred):
	y_pred_tags = getPredictions(y_pred)
	correct_pred = (y_pred_tags == y_test).float()
	acc = correct_pred.sum() / len(correct_pred)
	return acc

def selectModel(model_type):
	'''
	selects the model type and returns an object

	Args:
		model_type - model name as <str>
	Returns:
		model object
	Exception:
		-
	'''
	model_type = model_type.lower()
	if "bce" in model_type: loss_function = BCE_LOSS()
	elif "mssim" in model_type: loss_function = MS_SSIM_LOSS(data_range=1.0, size_average=True, channel=3)
	elif "sim" in model_type: loss_function = SSIM_LOSS(data_range=1.0, size_average=True, channel=3)
	elif "weight" in model_type: loss_function = WEIGHTED_SIMILARITY()
	elif "contractive" in model_type: loss_function = CONTRACTIVE_LOSS()
	else: loss_function = MSE_LOSS()
		
	if "variational" in model_type:
		if "base" in model_type: return VAE(), VARIATIONAL_LOSS()
		elif "kernel" in model_type: return KS_VAE(), VARIATIONAL_LOSS()
	elif "base" in model_type: return AE(), loss_function
	elif "q" in model_type: return Qiang_AutoEncoder(), loss_function
	elif "kernel" in model_type: return KS_AE(), loss_function
	elif "split" in model_type: return SM_AE(), loss_function
	else: raise TypeError("[ERROR]: Incorrect Model Type")

def selectData(dataset_type = "HAM10000", batch_size = 64):
	'''
	selects the dataset type and creates an object

	Args:
		dataset_type - dataset name as <str>
	Returns:
		dataset object
	Exception:
		-
	'''
	dataset_type = dataset_type.lower()
	if "ham" in dataset_type: return HAM10000_Dataset(batch_size = batch_size)
	elif "distraction" in dataset_type: return DISTRACTION_Dataset(batch_size = batch_size)
	else: raise TypeError("[ERROR]: Incorrect Dataset Type")
		
def getNParams(model):
	params_dict = dict()
	params_dict["Total"] = sum(p.numel() for p in model.parameters())
	params_dict["Trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
	params_dict["Non-Trainable"] = params_dict["Total"] - params_dict["Trainable"]
	return params_dict

def get_save_folder(model_type, data_type, version):
	model_version = "_".join([model_type, data_type, "V%d"%(version)])
	model_name = "_".join([model_type, data_type, "V%d.pth"%(version)])
	save_folder = os.path.join(Config.save_path, model_version)
	return model_name, save_folder

def create_path(model_type, data_type, version, createNew = True):
	# Path Configuration
	model_name, save_folder = get_save_folder(model_type, data_type, version)
	try: os.mkdir(save_folder)
	except:
		if createNew:
			version = max([int(d[-1]) for d in glob(save_folder[:-2] + "*")]) + 1
			model_name, save_folder = get_save_folder(model_type, data_type, version)
			os.mkdir(save_folder)
		else:
			pass
	return model_name, save_folder

def get_class_distribution(dataset_obj):
	count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
	for _, label_id in dataset_obj:
		label = idx2class[label_id]
		count_dict[label] += 1
	return count_dict