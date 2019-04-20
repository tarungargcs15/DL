import torch

CUDA = torch.cuda.is_available()

def save_checkpoint(path, model, epoch, loss):
	checkpoint = {}
	checkpoint["state_dict"] = model.state_dict()
	checkpoint["epoch"] = epoch
	checkpoint["loss"] = loss
	torch.save(checkpoint, path + "model.epoch%d" % epoch)

def Zeros(a, b, c):
	temp = torch.zeros(a, b, c)
	if CUDA:
		temp = temp.cuda()
	return temp

def LongTensor(a):
	temp = torch.LongTensor(a)
	if CUDA:
		temp = temp.cuda()
	return temp

def FloatTensor(a):
	temp = torch.FloatTensor(a)
	if CUDA:
		temp = temp.cuda()
	return temp

def padding(batch, pad_len, padding_vec):
	temp = [[padding_vec[i] for i in range(len(padding_vec))] for _ in range(pad_len - len(batch))]
	batch = batch + temp
	return batch