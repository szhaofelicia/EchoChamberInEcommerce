import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_rand_score


import pickle
from scipy import stats

from tqdm import tqdm
import json

np.random.seed(0)

def kmeans_clustering(x,k):
	clustering=KMeans(n_clusters=k,random_state=0).fit(x)
	y=clustering.labels_
	return clustering, y

"""
Following Group: Click Embedding

"""

with open('jsons/pos_user_click_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)
item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

nsamp=50 
ari_50=dict()
st=19
ed=29
ari=np.zeros((ed-st+1,nsamp))
for k in tqdm(range(st,ed+1)):
	idx=k-st
	print(k)
	for t in tqdm(range(nsamp)):
		x_samp,_=train_test_split(item_emb, test_size=0.512,random_state=t)#click:0.512 purchase:0.305
		x_train,_=train_test_split(x_samp, test_size=0.2,random_state=t) 
		fir_train=x_train[:,0,:]
		last_train=x_train[:,1,:]
		_,fir_y=kmeans_clustering(fir_train,k)
		_,last_y=kmeans_clustering(last_train,k)
		ari[idx][t]=adjusted_rand_score(fir_y,last_y)
ari_50['50']=ari
ari_50['average']=np.mean(ari,axis=1)


# for k in range(kmax-5,kmax+6):
for k in range(st,ed+1):
	idx=k-st
	print(k,'\t',ari[idx].mean())

with open("pickles/pos_click_ari_50_k%dto%d.pickle"%(st,ed),'wb') as file:
	pickle.dump(ari_50,file)


"""
Ignoring Group: Click Embedding

"""

with open('jsons/neg_user_click_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)
item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

nsamp=50 
ari_50=dict()
st=15
ed=25
ari=np.zeros((ed-st+1,nsamp))
for k in tqdm(range(st,ed+1)):
	idx=k-st
	print(k)
	for t in tqdm(range(nsamp)):
		x_train,_=train_test_split(item_emb, test_size=0.2,random_state=t) #
		fir_train=x_train[:,0,:]
		last_train=x_train[:,1,:]
		_,fir_y=kmeans_clustering(fir_train,k)
		_,last_y=kmeans_clustering(last_train,k)
		ari[idx][t]=adjusted_rand_score(fir_y,last_y)
ari_50['50']=ari
ari_50['average']=np.mean(ari,axis=1)


# for k in range(kmax-5,kmax+6):
for k in range(st,ed+1):
	idx=k-st
	print(k,'\t',ari[idx].mean())

with open("pickles/neg_click_ari_50_k%dto%d.pickle"%(st,ed),'wb') as file:
	pickle.dump(ari_50,file)


"""
Following Group: Purchase Embedding

"""

with open('jsons/pos_user_purchase_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)
item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

nsamp=50 
ari_50=dict()
st=6
ed=16
ari=np.zeros((ed-st+1,nsamp))
for k in tqdm(range(st,ed+1)):
	idx=k-st
	print(k)
	for t in tqdm(range(nsamp)):
		x_samp,_=train_test_split(item_emb, test_size=0.305,random_state=t)#click:0.512 purchase:0.305
		x_train,_=train_test_split(x_samp, test_size=0.2,random_state=t) 
		fir_train=x_train[:,0,:]
		last_train=x_train[:,1,:]
		_,fir_y=kmeans_clustering(fir_train,k)
		_,last_y=kmeans_clustering(last_train,k)
		ari[idx][t]=adjusted_rand_score(fir_y,last_y)
ari_50['50']=ari
ari_50['average']=np.mean(ari,axis=1)


# for k in range(kmax-5,kmax+6):
for k in range(st,ed+1):
	idx=k-st
	print(k,'\t',ari[idx].mean())

with open("pickles/pos_purchase_ari_50_k%dto%d.pickle"%(st,ed),'wb') as file:
	pickle.dump(ari_50,file)


"""
Ignoring Group: Purchase Embedding

"""

with open('jsons/neg_user_purchase_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)
item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

nsamp=50 
ari_50=dict()
st=4
ed=14
ari=np.zeros((ed-st+1,nsamp))
for k in tqdm(range(st,ed+1)):
	idx=k-st
	print(k)
	for t in tqdm(range(nsamp)):
		x_train,_=train_test_split(item_emb, test_size=0.2,random_state=t) #
		fir_train=x_train[:,0,:]
		last_train=x_train[:,1,:]
		_,fir_y=kmeans_clustering(fir_train,k)
		_,last_y=kmeans_clustering(last_train,k)
		ari[idx][t]=adjusted_rand_score(fir_y,last_y)
ari_50['50']=ari
ari_50['average']=np.mean(ari,axis=1)


# for k in range(kmax-5,kmax+6):
for k in range(st,ed+1):
	idx=k-st
	print(k,'\t',ari[idx].mean())

with open("pickles/neg_purchase_ari_50_k%dto%d.pickle"%(st,ed),'wb') as file:
	pickle.dump(ari_50,file)
