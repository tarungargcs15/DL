import numpy as np
import os
import nltk

BATCH_SIZE = 4096
EMBEDDING_SIZE = 300

path = 'dataset/Batches/' + str(BATCH_SIZE)

if not (os.path.isdir(path)):
	os.mkdir(path)
	os.mkdir(path + '/train')
	os.mkdir(path + '/test')
	os.mkdir(path + '/train/q1')
	os.mkdir(path + '/train/q2')
	os.mkdir(path + '/train/is_dup')
	os.mkdir(path + '/test/q1')
	os.mkdir(path + '/test/q2')
	os.mkdir(path + '/test/test_id')

def find_embedding(question, vocab_embed):
	embed = []
	for word in question:
		embed.append(vocab_embed.get(word, [0.0] * EMBEDDING_SIZE))
	return embed

def create_batches_training(file, path):
	vocab_embed = np.load('dataset/Vocab_Embedding.npy').tolist()
	print('Vocabulary Embedding Loaded')

	data1 = np.load(file+'q1.npy').tolist()
	data2 = np.load(file+'q2.npy').tolist()
	data3 = np.load(file+'is_dup.npy').tolist()
	print('Data Files Loaded')
	
	batch_count = 0
	batch1 = []
	batch_embed1 = []
	batch2 = []
	batch_embed2 = []
	batch3 = []
	print('No. of Batches created:')
	for i in range(len(data1)):
		if len(batch1) == BATCH_SIZE:
			batch_count += 1
			np.save(path + 'q1/batch' + str(batch_count) + '.npy', batch1)
			np.save(path + 'q1/batch_embed' + str(batch_count) + '.npy', batch_embed1)
			np.save(path+'q2/batch' + str(batch_count) + '.npy', batch2)
			np.save(path + 'q2/batch_embed' + str(batch_count) + '.npy', batch_embed2)
			np.save(path+'is_dup/batch' + str(batch_count) + '.npy', batch3)
			batch1 = []
			batch_embed1 = []
			batch2 = []
			batch_embed2 = []
			batch3 = []
			print(batch_count, end = ' ', flush = True)

		batch1.append(data1[i])
		batch_embed1.append(find_embedding(data1[i], vocab_embed))
		batch2.append(data2[i])
		batch_embed2.append(find_embedding(data2[i], vocab_embed))
		batch3.append(data3[i])

	if(len(batch1) != 0):
		batch_count += 1
		np.save(path + 'q1/batch' + str(batch_count) + '.npy', batch1)
		np.save(path + 'q1/batch_embed' + str(batch_count) + '.npy', batch_embed1)
		np.save(path+'q2/batch' + str(batch_count) + '.npy', batch2)
		np.save(path + 'q2/batch_embed' + str(batch_count) + '.npy', batch_embed2)
		np.save(path+'is_dup/batch' + str(batch_count) + '.npy', batch3)
		print(batch_count)

def create_batches_testing(file, path):
	data1 = np.load(file + 'q1.npy').tolist()
	data2 = np.load(file + 'q2.npy').tolist()
	data3 = np.load(file + 'ids.npy').tolist()
	print('Data Files Loaded')
	
	batch_count = 0
	batch1 = []
	batch2 = []
	batch3 = []
	print('No. of Batches created:')
	for i in range(len(data1)):
		if len(batch1) == BATCH_SIZE:
			batch_count += 1
			np.save(path+'q1/' + '/batch' + str(batch_count) + '.npy', batch1)
			np.save(path+'q2/' + '/batch' + str(batch_count) + '.npy', batch2)
			np.save(path+'test_id/' + '/batch' + str(batch_count) + '.npy', batch3)
			batch1 = []
			batch2 = []
			batch3 = []
			print(batch_count, end = ' ', flush = True)

		batch1.append(data1[i])
		batch2.append(data2[i])
		batch3.append(data3[i])

	if(len(batch1) != 0):
		batch_count += 1
		np.save(path+'q1/batch' + str(batch_count) + '.npy', batch1)
		np.save(path+'q2/batch' + str(batch_count) + '.npy', batch2)
		np.save(path+'test_id/batch' + str(batch_count) + '.npy', batch3)
		print(batch_count)

print('Creating Batches of Training data')
create_batches_training('dataset/train_', path + '/train/')

print('\nCreating Batches of Testing data')
create_batches_testing('dataset/test_', path + '/test/')


