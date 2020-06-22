import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_chunked

import pickle
from scipy import stats

from tqdm import tqdm
import json

np.random.seed(0)


"""
Following group
"""
with open('jsons/pos_user_display_diversity.json','r') as load_f:
    user_display = json.load(load_f)
pos_div=np.array(list(user_display.values())) 
pos_samp, _ = train_test_split(pos_div, test_size=0.6531,random_state=0) # display

fir_div=pos_samp[:,0] 
last_div=pos_samp[:,1] 

print('Average diversity in first block','Average diversity in last block')
print(fir_div.mean(),last_div.mean()) 


## test of normal distribution
kstest_=stats.kstest(fir_div, 'norm') #(D,p-value)
anderson_=stats.anderson(fir_div,dist='norm')

print('Diversity in first block')
print(kstest_) 
print(anderson_)

kstest_=stats.kstest(last_div, 'norm') #(D,p-value)
anderson_=stats.anderson(last_div,dist='norm')

print('Diversity in last block')
print(kstest_) 
print(anderson_)


## T-test
levene_p=stats.levene(fir_div, last_div) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(fir_div,last_div,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(fir_div,last_div,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)


"""
Ignoring grup
"""
with open('jsons/neg_user_display_diversity.json','r') as load_f:
    user_display = json.load(load_f) 

neg_div=np.array(list(user_display.values())) 

fir_div=neg_div[:,0] 
last_div=neg_div[:,1] 

print('Average diversity in first block','Average diversity in last block')
print(fir_div.mean(),last_div.mean()) 


## test of normal distribution
kstest_=stats.kstest(fir_div, 'norm') #(D,p-value)
anderson_=stats.anderson(fir_div,dist='norm')

print('Diversity in first block')
print(kstest_) 
print(anderson_)

kstest_=stats.kstest(last_div, 'norm') #(D,p-value)
anderson_=stats.anderson(last_div,dist='norm')

print('Diversity in last block')
print(kstest_) 
print(anderson_)


## T-test
levene_p=stats.levene(fir_div, last_div) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(fir_div,last_div,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(fir_div,last_div,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)


"""
All group
"""
all_div=np.concatenate([pos_samp,neg_div],axis=0)

fir_div=all_div[:,0] 
last_div=all_div[:,1] 

print('Average diversity in first block','Average diversity in last block')
print(fir_div.mean(),last_div.mean()) 


## test of normal distribution
kstest_=stats.kstest(fir_div, 'norm') #(D,p-value)
anderson_=stats.anderson(fir_div,dist='norm')

print('Diversity in first block')
print(kstest_) 
print(anderson_)

kstest_=stats.kstest(last_div, 'norm') #(D,p-value)
anderson_=stats.anderson(last_div,dist='norm')

print('Diversity in last block')
print(kstest_) 
print(anderson_)


## T-test
levene_p=stats.levene(fir_div, last_div) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(fir_div,last_div,equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(fir_div,last_div,equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)



"""
Between groups
"""

## Tests of normal distribution have been completed in the previous steps

## T-test
# First block
levene_p=stats.levene(pos_samp[:,0], neg_div[:,0]) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(pos_samp[:,0],neg_div[:,0],equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(pos_samp[:,0],neg_div[:,0],equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)


# Last block
levene_p=stats.levene(pos_samp[:,1], neg_div[:,1]) # equal_var or not
print(levene_p)

t,p=stats.ttest_ind(pos_samp[:,1],neg_div[:,1],equal_var=True) # Null hypothesis:equal
print('Equal distribution',p)
t,p=stats.ttest_ind(pos_samp[:,1],neg_div[:,1],equal_var=False) # Null hypothesis: not equal
print('Not equal distribution',p)