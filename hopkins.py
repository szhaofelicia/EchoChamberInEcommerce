import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy import stats
from pyclustertend import hopkins

from tqdm import tqdm
import json
import pickle


with open('../jsons/pos_user_click_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)

item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

hopkins_dict=dict()
h_fir=np.zeros((50,))
h_last=np.zeros((50,))
for t in range(50):
	x_samp, _ = train_test_split(item_emb, test_size=0.512,random_state=t) #click:0.512 purchase:0.305
	fir_list=x_samp[:,0,:] # len * 128
	last_list=x_samp[:,1,:] # len * 128
	h_fir[t]=1-hopkins(fir_list,int(0.1 *len(fir_list))) #h=0.5,x; x
	h_last[t]=1-hopkins(last_list,int(0.1 *len(last_list))) #h=0.5,x; x
hopkins_dict['first']=h_fir
hopkins_dict['last']=h_last
hopkins_dict['first average']=np.mean(h_fir)
hopkins_dict['last average']=np.mean(h_last)
print('50',np.mean(h_fir),np.mean(h_last))
with open("../dict_pickles/afBIC/pos_click_hopkins_50.pickle",'wb') as file:
	pickle.dump(hopkins_dict,file)

# Statistics of Hopkins

with open('../dict_pickles/afBIC/pos_purchase_hopkins_50.pickle', 'rb') as f: # include char and title(item)
    hopkins_dict=pickle.load(f)
pos_fir=hopkins_dict['first'] .reshape(50,1)
pos_last=hopkins_dict['last'] .reshape(50,1)
pos_all=np.concatenate([pos_fir,pos_last],axis=1)


with open('../dict_pickles/afBIC/neg_purchase_hopkins_50.pickle', 'rb') as f: # include char and title(item)
    hopkins_dict=pickle.load(f)
neg_fir=hopkins_dict['first'] .reshape(50,1)
neg_last=hopkins_dict['last'] .reshape(50,1)
neg_all=np.concatenate([neg_fir,neg_last],axis=1)

all_hopkins=np.concatenate([pos_all,neg_all],axis=0)
all_fir=all_hopkins[:,0]
all_last=all_hopkins[:,1]

print(all_fir.mean(),all_last.mean())

## test of normal distribution
kstest_=stats.kstest(all_fir, 'norm') #(D,p-value)
anderson_=stats.anderson(all_fir,dist='norm')

print('positive')
print(kstest_) 
print(anderson_)


kstest_=stats.kstest(all_last, 'norm') #(D,p-value)
anderson_=stats.anderson(all_last,dist='norm')
print('negative')
print(kstest_) 
print(anderson_)


levene_p=stats.levene(all_fir, all_last) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(all_fir,all_last,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(all_fir,all_last,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)