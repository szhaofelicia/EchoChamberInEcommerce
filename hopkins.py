import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy import stats
from pyclustertend import hopkins

from tqdm import tqdm
import json
import pickle

np.random.seed(0)

# Measure Hopkins Statistic

"""
Following Group: Click Embedding

"""

with open('jsons/pos_user_click_embed.json','r') as load_f:
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
# with open("pickles/pos_click_hopkins_50.pickle",'wb') as file:
# 	pickle.dump(hopkins_dict,file)

"""
Ignoring Group: Click Embedding

"""

with open('jsons/neg_user_click_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)

item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

hopkins_dict=dict()
h_fir=np.zeros((50,))
h_last=np.zeros((50,))
for t in range(50):
	fir_list=item_emb[:,0,:] # len * 128
	last_list=item_emb[:,1,:] # len * 128
	h_fir[t]=1-hopkins(fir_list,int(0.1 *len(fir_list))) #h=0.5,x; x
	h_last[t]=1-hopkins(last_list,int(0.1 *len(last_list))) #h=0.5,x; x
hopkins_dict['first']=h_fir
hopkins_dict['last']=h_last
hopkins_dict['first average']=np.mean(h_fir)
hopkins_dict['last average']=np.mean(h_last)
print('50',np.mean(h_fir),np.mean(h_last))
# with open("pickles/neg_click_hopkins_50.pickle",'wb') as file:
# 	pickle.dump(hopkins_dict,file)
"""
Following Group: Purchase Embedding

"""

with open('jsons/pos_user_purchase_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)

item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

hopkins_dict=dict()
h_fir=np.zeros((50,))
h_last=np.zeros((50,))
for t in range(50):
	x_samp, _ = train_test_split(item_emb, test_size=0.305,random_state=t) #click:0.512 purchase:0.305
	fir_list=x_samp[:,0,:] # len * 128
	last_list=x_samp[:,1,:] # len * 128
	h_fir[t]=1-hopkins(fir_list,int(0.1 *len(fir_list))) #h=0.5,x; x
	h_last[t]=1-hopkins(last_list,int(0.1 *len(last_list))) #h=0.5,x; x
hopkins_dict['first']=h_fir
hopkins_dict['last']=h_last
hopkins_dict['first average']=np.mean(h_fir)
hopkins_dict['last average']=np.mean(h_last)
print('50',np.mean(h_fir),np.mean(h_last))
# with open("pickles/pos_purchase_hopkins_50.pickle",'wb') as file:
# 	pickle.dump(hopkins_dict,file)

"""
Ignoring Group: Purchase Embedding

"""

with open('jsons/neg_user_purchase_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)

item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128

hopkins_dict=dict()
h_fir=np.zeros((50,))
h_last=np.zeros((50,))
for t in range(50):
	fir_list=item_emb[:,0,:] # len * 128
	last_list=item_emb[:,1,:] # len * 128
	h_fir[t]=1-hopkins(fir_list,int(0.1 *len(fir_list))) 
	h_last[t]=1-hopkins(last_list,int(0.1 *len(last_list))) #
hopkins_dict['first']=h_fir
hopkins_dict['last']=h_last
hopkins_dict['first average']=np.mean(h_fir)
hopkins_dict['last average']=np.mean(h_last)
print('50',np.mean(h_fir),np.mean(h_last))
# with open("pickles/neg_purchase_hopkins_50.pickle",'wb') as file:
# 	pickle.dump(hopkins_dict,file)


#################################################################################################
# Statistic Significance and p-value

"""
Click Embedding
"""

with open('pickles/pos_click_hopkins_50.pickle', 'rb') as f: # include char and title(item)
    hopkins_dict=pickle.load(f)
pos_fir=hopkins_dict['first'] 
pos_last=hopkins_dict['last'] 
pos_all=np.concatenate([pos_fir.reshape(50,1),pos_last.reshape(50,1)],axis=1) 


with open('pickles/neg_click_hopkins_50.pickle', 'rb') as f: # include char and title(item)
    hopkins_dict=pickle.load(f)
neg_fir=hopkins_dict['first']
neg_last=hopkins_dict['last']
neg_all=np.concatenate([neg_fir.reshape(50,1),neg_last.reshape(50,1)],axis=1)# 50*2

all_hopkins=np.concatenate([pos_all,neg_all],axis=0) # 100*2
all_fir=all_hopkins[:,0] # 100
all_last=all_hopkins[:,1] # 100

### ALL Group p-value

print(all_fir.mean(),all_last.mean())

## test of normal distribution
kstest_=stats.kstest(all_fir, 'norm') #(D,p-value)
anderson_=stats.anderson(all_fir,dist='norm')

print('First block')
print(kstest_) 
print(anderson_)


kstest_=stats.kstest(all_last, 'norm') #(D,p-value)
anderson_=stats.anderson(all_last,dist='norm')
print('Last block')
print(kstest_) 
print(anderson_)


levene_p=stats.levene(all_fir, all_last) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(all_fir,all_last,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(all_fir,all_last,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)


### Folllowing Group p-value

print(pos_fir.mean(),pos_last.mean())

## test of normal distribution
kstest_=stats.kstest(pos_fir, 'norm') #(D,p-value)
anderson_=stats.anderson(pos_fir,dist='norm')

print('First block')
print(kstest_) 
print(anderson_)


kstest_=stats.kstest(pos_last, 'norm') #(D,p-value)
anderson_=stats.anderson(pos_last,dist='norm')
print('Last block')
print(kstest_) 
print(anderson_)


levene_p=stats.levene(pos_fir, pos_last) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(pos_fir,pos_last,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(pos_fir,pos_last,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)

### Ignoring Group p-value

print(neg_fir.mean(),neg_last.mean())

## test of normal distribution
kstest_=stats.kstest(neg_fir, 'norm') #(D,p-value)
anderson_=stats.anderson(neg_fir,dist='norm')

print('First block')
print(kstest_) 
print(anderson_)


kstest_=stats.kstest(neg_last, 'norm') #(D,p-value)
anderson_=stats.anderson(neg_last,dist='norm')
print('Last block')
print(kstest_) 
print(anderson_)


levene_p=stats.levene(neg_fir, neg_last) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(neg_fir,neg_last,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(neg_fir,neg_last,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)


"""
Click Embedding: between-group p-value
"""
### Between Group p-value: first block

print(pos_fir.mean(),neg_fir.mean())


levene_p=stats.levene(pos_fir, neg_fir) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(pos_fir,neg_fir,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(pos_fir,neg_fir,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)

### Between Group p-value: last block

print(pos_last.mean(),neg_last.mean())


levene_p=stats.levene(pos_last, neg_last) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(pos_last,neg_last,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(pos_last,neg_last,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)


#####################################################################################

"""
Purchase Embedding
"""

with open('pickles/pos_purchase_hopkins_50.pickle', 'rb') as f: # include char and title(item)
    hopkins_dict=pickle.load(f)
pos_fir=hopkins_dict['first'] 
pos_last=hopkins_dict['last'] 
pos_all=np.concatenate([pos_fir.reshape(50,1),pos_last.reshape(50,1)],axis=1) 


with open('pickles/neg_purchase_hopkins_50.pickle', 'rb') as f: # include char and title(item)
    hopkins_dict=pickle.load(f)
neg_fir=hopkins_dict['first'] 
neg_last=hopkins_dict['last']
neg_all=np.concatenate([neg_fir.reshape(50,1),neg_last.reshape(50,1)],axis=1)# 50*2

all_hopkins=np.concatenate([pos_all,neg_all],axis=0) # 100*2
all_fir=all_hopkins[:,0] # 100
all_last=all_hopkins[:,1] # 100

### ALL Group p-value

print(all_fir.mean(),all_last.mean())

## test of normal distribution
kstest_=stats.kstest(all_fir, 'norm') #(D,p-value)
anderson_=stats.anderson(all_fir,dist='norm')

print('First block')
print(kstest_) 
print(anderson_)


kstest_=stats.kstest(all_last, 'norm') #(D,p-value)
anderson_=stats.anderson(all_last,dist='norm')
print('Last block')
print(kstest_) 
print(anderson_)


levene_p=stats.levene(all_fir, all_last) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(all_fir,all_last,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(all_fir,all_last,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)


### Folllowing Group p-value

print(pos_fir.mean(),pos_last.mean())

## test of normal distribution
kstest_=stats.kstest(pos_fir, 'norm') #(D,p-value)
anderson_=stats.anderson(pos_fir,dist='norm')

print('First block')
print(kstest_) 
print(anderson_)


kstest_=stats.kstest(pos_last, 'norm') #(D,p-value)
anderson_=stats.anderson(pos_last,dist='norm')
print('Last block')
print(kstest_) 
print(anderson_)


levene_p=stats.levene(pos_fir, pos_last) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(pos_fir,pos_last,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(pos_fir,pos_last,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)

### Ignoring Group p-value

print(neg_fir.mean(),neg_last.mean())

## test of normal distribution
kstest_=stats.kstest(neg_fir, 'norm') #(D,p-value)
anderson_=stats.anderson(neg_fir,dist='norm')

print('First block')
print(kstest_) 
print(anderson_)


kstest_=stats.kstest(neg_last, 'norm') #(D,p-value)
anderson_=stats.anderson(neg_last,dist='norm')
print('Last block')
print(kstest_) 
print(anderson_)


levene_p=stats.levene(neg_fir, neg_last) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(neg_fir,neg_last,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(neg_fir,neg_last,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)


"""
Purchases Embedding: between-group p-value
"""
### Between Group p-value: first block

print(pos_fir.mean(),neg_fir.mean())


levene_p=stats.levene(pos_fir, neg_fir) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(pos_fir,neg_fir,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(pos_fir,neg_fir,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)

### Between Group p-value: last block

print(pos_last.mean(),neg_last.mean())


levene_p=stats.levene(pos_last, neg_last) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(pos_last,neg_last,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(pos_last,neg_last,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)



