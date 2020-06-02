import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import calinski_harabasz_scores

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
ch_50=dict()
st=19
ed=29
ch_f=c.zeros((ed-st+1,nsamp))
ch_l=c.zeros((ed-st+1,nsamp))

for k in tqdm(range(st,ed+1)):
	idx=k-st
	for t in tqdm(range(nsamp)):
		x_samp,_=train_test_split(item_emb, test_size=0.512,random_state=t)#click:0.512 purchase:0.305
		x_train,_=train_test_split(x_samp, test_size=0.2,random_state=t) 
		fir_train=x_train[:,0,:]
		last_train=x_train[:,1,:]
		_,y=kmeans_clustering(fir_train,k)
		ch_f[idx][t]=calinski_harabasz_score(fir_train,y)
		ch_l[idx][t]=calinski_harabasz_score(last_train,y)
ch_50['first']=ch_f
ch_50['last']=ch_l
ch_50['first average']=np.mean(ch_f,axis=1)
ch_50['last average']=np.mean(ch_l,axis=1)

for k in range(st,ed+1):
	idx=k-st
	print(k,'\t',ch_f[idx].mean(),'\t',ch_l[idx].mean())

with open("pickles/pos_click_ch_50_k%dto%d.pickle"%(st,ed),'wb') as file:
	pickle.dump(ch_50,file)


"""
Ignoring Group: Click Embedding

"""

with open('jsons/neg_user_click_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)
item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

nsamp=50 
ch_50=dict()
st=15
ed=25
ch_f=c.zeros((ed-st+1,nsamp))
ch_l=c.zeros((ed-st+1,nsamp))

for k in tqdm(range(st,ed+1)):
	idx=k-st
	for t in tqdm(range(nsamp)):
		x_train,_=train_test_split(item_emb, test_size=0.2,random_state=t) 
		fir_train=x_train[:,0,:]
		last_train=x_train[:,1,:]
		_,y=kmeans_clustering(fir_train,k)
		ch_f[idx][t]=calinski_harabasz_score(fir_train,y)
		ch_l[idx][t]=calinski_harabasz_score(last_train,y)
ch_50['first']=ch_f
ch_50['last']=ch_l
ch_50['first average']=np.mean(ch_f,axis=1)
ch_50['last average']=np.mean(ch_l,axis=1)

for k in range(st,ed+1):
	idx=k-st
	print(k,'\t',ch_f[idx].mean(),'\t',ch_l[idx].mean())

with open("pickles/neg_click_ch_50_k%dto%d.pickle"%(st,ed),'wb') as file:
	pickle.dump(ch_50,file)


"""
Following Group: Purchase Embedding

"""

with open('jsons/pos_user_purchase_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)
item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

nsamp=50 
ch_50=dict()
st=6
ed=16
ch_f=c.zeros((ed-st+1,nsamp))
ch_l=c.zeros((ed-st+1,nsamp))

for k in tqdm(range(st,ed+1)):
	idx=k-st
	for t in tqdm(range(nsamp)):
		x_samp,_=train_test_split(item_emb, test_size=0.305,random_state=t)#click:0.512 purchase:0.305
		x_train,_=train_test_split(x_samp, test_size=0.2,random_state=t) 
		fir_train=x_train[:,0,:]
		last_train=x_train[:,1,:]
		_,y=kmeans_clustering(fir_train,k)
		ch_f[idx][t]=calinski_harabasz_score(fir_train,y)
		ch_l[idx][t]=calinski_harabasz_score(last_train,y)
ch_50['first']=ch_f
ch_50['last']=ch_l
ch_50['first average']=np.mean(ch_f,axis=1)
ch_50['last average']=np.mean(ch_l,axis=1)

for k in range(st,ed+1):
	idx=k-st
	print(k,'\t',ch_f[idx].mean(),'\t',ch_l[idx].mean())

with open("pickles/pos_purchase_ch_50_k%dto%d.pickle"%(st,ed),'wb') as file:
	pickle.dump(ch_50,file)


"""
Ignoring Group: Purchase Embedding

"""

with open('jsons/neg_user_purchase_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)
item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

nsamp=50 
ch_50=dict()
st=4
ed=14
ch_f=c.zeros((ed-st+1,nsamp))
ch_l=c.zeros((ed-st+1,nsamp))

for k in tqdm(range(st,ed+1)):
	idx=k-st
	for t in tqdm(range(nsamp)):
		x_samp,_=train_test_split(item_emb, test_size=0.512,random_state=t)#click:0.512 purchase:0.305
		x_train,_=train_test_split(x_samp, test_size=0.2,random_state=t) 
		fir_train=x_train[:,0,:]
		last_train=x_train[:,1,:]
		_,y=kmeans_clustering(fir_train,k)
		ch_f[idx][t]=calinski_harabasz_score(fir_train,y)
		ch_l[idx][t]=calinski_harabasz_score(last_train,y)
ch_50['first']=ch_f
ch_50['last']=ch_l
ch_50['first average']=np.mean(ch_f,axis=1)
ch_50['last average']=np.mean(ch_l,axis=1)

for k in range(st,ed+1):
	idx=k-st
	print(k,'\t',ch_f[idx].mean(),'\t',ch_l[idx].mean())

with open("pickles/neg_purchase_ch_50_k%dto%d.pickle"%(st,ed),'wb') as file:
	pickle.dump(ch_50,file)


