import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from model import *

NO_TRAIN_BATCHES = 198#395#198
NO_TEST_BATCHES = 1146#2291#1146

path = 'dataset/Batches/' + str(BATCH_SIZE) + '/'
def to_embed(data):
	vocab_embed = np.load('dataset/Vocab_Embedding.npy').tolist()
	batch = []
	len_array = []
	for sentence in data:
		sentence_embed = []
		for word in sentence:
			sentence_embed.append(vocab_embed.get(word, [0]*EMBEDDING_SIZE))
		len_array.append(min(len(sentence), PADDING_LENGTH))
		batch.append(padding(sentence_embed, PADDING_LENGTH, [0]*300)[:PADDING_LENGTH])
	return batch, len_array

def pad(batch):
	padded_batch = []
	len_array = []
	for question in batch:
		len_array.append(min(len(question), PADDING_LENGTH))
		padded_batch.append(padding(question, PADDING_LENGTH, [0]*300)[:PADDING_LENGTH])
	return padded_batch, len_array

def test(model):
	counter = 0
	f = open('results/' + str(BATCH_SIZE) + '/test.csv', 'w')

	print("Number of batches, model tested for:")
	for i in range(1, NO_TEST_BATCHES+1):
		counter += 1
		q1_batch, q1_len = to_embed(np.load(path + 'test/q1/batch' + str(i) + '.npy').tolist())
		q2_batch, q2_len = to_embed(np.load(path + 'test/q2/batch' + str(i) + '.npy').tolist())
		test_ids = np.load(path + 'test/test_id/batch' + str(i) + '.npy').tolist()
		_, predictions = model(q1_batch, q2_batch, q1_len, q2_len, False)
		for ids, prediction in zip(test_ids, predictions):
			f.write("%d,%d\n"%(ids, prediction))
		print(counter, end = " ", flush = True)

	f.close()


def train():
	model = dl_model()
	optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

	for ei in range(1, EPOCH+1):	
		loss_sum = 0
		counter = 0

		print('Epoch: ', ei)
		print("Number of batches, model trained for:")
		for i in range(1, NO_TRAIN_BATCHES+1):
			q1_batch, q1_len = pad(np.load(path + 'train/q1/batch_embed' + str(i) + '.npy').tolist())
			q2_batch, q2_len = pad(np.load(path + 'train/q2/batch_embed' + str(i) + '.npy').tolist())
			target_batch = np.load(path + 'train/is_dup/batch' + str(i) + '.npy').tolist()

			for i in range(len(target_batch)):
				if target_batch[i] == 0:
					target_batch[i] = -1 
			model.zero_grad()
			loss, _ = model(q1_batch, q2_batch, q1_len, q2_len, True, target_batch)
			loss.backward()
			optimizer.step()
			loss_sum += float(loss.data)
			counter += 1
			print(counter, end = " ", flush = True)

		save_checkpoint('results/' + str(BATCH_SIZE) + '/', model, ei, loss_sum)
		print("\nLoss:", loss_sum)
	return model
	
if __name__ == '__main__':
	model = train()
	test(model)
