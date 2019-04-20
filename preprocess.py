import pandas as pd
import numpy as np
import nltk

def preprocess(sentence):
	if(type(sentence) == float):
		return []
	words = nltk.word_tokenize(sentence)
	return words
'''
df_train = pd.read_csv('dataset/train.csv')

df_train1 = df_train[['id', 'question1']]
df_train2 = df_train[['id', 'question2']]
df_train3 = df_train[['id', 'is_duplicate']]
dict1 = df_train1.set_index('id')['question1'].to_dict()
dict2 = df_train2.set_index('id')['question2'].to_dict()
dict3 = df_train3.set_index('id')['is_duplicate'].to_dict()

questions1 = []
questions2 = []
is_dup = []
for key in dict1.keys():
	questions1.append(preprocess(dict1[key]))
	questions2.append(preprocess(dict2[key]))
	is_dup.append(dict3[key])
np.save('dataset/train_q1.npy', questions1)
np.save('dataset/train_q2.npy', questions2)
np.save('dataset/train_is_dup.npy', is_dup)
'''


df_test = pd.read_csv('dataset/test.csv')

df_test1 = df_test[['test_id', 'question1']]
df_test2 = df_test[['test_id', 'question2']]

dict1 = df_test1.set_index('test_id')['question1'].to_dict()
dict2 = df_test2.set_index('test_id')['question2'].to_dict()

questions1 = []
questions2 = []
test_ids = []
for key in dict1.keys():
	questions1.append(preprocess(dict1[key]))
	questions2.append(preprocess(dict2[key]))
	test_ids.append(key)
np.save('dataset/test_q1.npy', questions1)
np.save('dataset/test_q2.npy', questions2)
np.save('dataset/test_ids.npy', test_ids)
