import numpy as np
import pandas as pd

from random import shuffle,sample

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_rand_score


import pickle
from scipy import stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
import json

np.random.seed(0)

def kmeans_clustering(x,k):
	clustering=KMeans(n_clusters=k,random_state=0,max_iter=1000).fit(x)
	y=clustering.labels_
	return clustering, y

np.random.seed(0)

with open('jsons/pos_user_click_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)
pos_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

nsamp=50 #200
ari_50=dict()
st=19
ed=29
ari=np.zeros((ed-st+1,nsamp))
for k in tqdm(range(st,ed+1)):
	idx=k-st
	print(k)
	for t in tqdm(range(nsamp)):
		x_samp,_=train_test_split(pos_emb, test_size=0.512,random_state=t)#click:0.512 purchase:0.305
		x_train,_=train_test_split(x_samp, test_size=0.2,random_state=t) #
		# print(x_samp.shape,x_train.shape)
		fir_train=x_train[:,0,:]
		last_train=x_train[:,1,:]
		_,fir_y=kmeans_clustering(fir_train,k)
		_,last_y=kmeans_clustering(last_train,k)
		ari[idx][t]=adjusted_rand_score(fir_y,last_y)
		# print(t,ch_f[k-2][t],ch_l[k-2][t])
ari_50['50']=ari
ari_50['average']=np.mean(ari,axis=1)