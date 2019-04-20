import torch
import torch.nn as nn
from utils import *

EPOCH = 10
PADDING_LENGTH = 100
EMBEDDING_SIZE = 300
BATCH_SIZE = 2048

NUM_LAYERS = 2
HIDDEN_SIZE = 100
BIDIRECTIONAL = False
NUM_DIRS = 2 if BIDIRECTIONAL else 1
LEARNING_RATE = 0.001
DROPOUT = 0.45

SIM_THRESHOLD = 0.5

class dl_model(nn.Module):
	def __init__(self):
		super().__init__()
		self.lstm = nn.LSTM(input_size = EMBEDDING_SIZE,
							hidden_size = HIDDEN_SIZE // NUM_DIRS,
							num_layers = NUM_LAYERS,
							batch_first = True, 
							bidirectional = BIDIRECTIONAL,
							dropout = DROPOUT)
		self.loss_fn = nn.CosineEmbeddingLoss()
		self.sim_metric = nn.CosineSimilarity(dim = 1)
		if CUDA:
			self.cuda()
	
	def init_hidden(self, input_batch):
		batch_size = input_batch.size()[0]
		a = Zeros(NUM_LAYERS * NUM_DIRS, batch_size, HIDDEN_SIZE // NUM_DIRS)
		b = Zeros(NUM_LAYERS * NUM_DIRS, batch_size, HIDDEN_SIZE // NUM_DIRS)
		return (a,b)

	def forward(self, input_batch1, input_batch2, len1, len2, is_training, target_batch = None):
		input_batch1 = FloatTensor(input_batch1)
		input_batch2 = FloatTensor(input_batch2)
		hidden1 = self.init_hidden(input_batch1)
		h1, _ = self.lstm(input_batch1, hidden1)
		hidden2 = self.init_hidden(input_batch2)
		h2, _ = self.lstm(input_batch2, hidden2)
		q1_batch = []
		q2_batch = []
		for i in range(input_batch1.size()[0]):
			q1_batch.append(h1[i][len1[i]-1])
			q2_batch.append(h2[i][len2[i]-1])
		q1_batch = torch.stack(q1_batch)
		q2_batch = torch.stack(q2_batch)
		if is_training:
			target_batch = FloatTensor(target_batch)
			return self.loss_fn(q1_batch, q2_batch, target_batch), None
		else:
			sim_scores = self.sim_metric(q1_batch, q2_batch)
			predictions = []
			for i in range(input_batch1.size()[0]):
				if sim_scores[i] >= SIM_THRESHOLD:
					predictions.append(1)
				else:
					predictions.append(0)
			return None, predictions
