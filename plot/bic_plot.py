import numpy as np
import matplotlib.pyplot as plt

import json
import pickle

# Click embedding in Following group
with open("pickles/pos_click_bic_50.pickle",'rb') as file:
    bic=pickle.load(file)

"""
# Click embedding in Ignoring group
with open("pickles/neg_click_bic_50.pickle",'rb') as file:
    bic=pickle.load(file)

# Purchase embedding in Following group
with open("pickles/pos_purchase_bic_50.pickle",'rb') as file:
    bic=pickle.load(file)

# Purchase embedding in Ignoring group
with open("pickles/neg_purchase_bic_50.pickle",'rb') as file:
    bic=pickle.load(file)
"""

kmax=50
kopt=24 # pos_purchase: 11, pos_click:24, neg_purchase:9 ,neg_click:20
ks=np.array(range(2,kmax+1))

plt.figure(figsize=(8,6))  
plt.plot(ks,np.mean(bic,axis=1),marker='*',mfc='xkcd:azure',linestyle='-',color='xkcd:azure')
plt.legend(['BIC'])
plt.axvline(x=kopt, c='xkcd:silver', ls='-.', lw=2)
plt.xlabel("Number of clusters",fontsize=16)
plt.ylabel("Bayesian Information Criterion",fontsize=16)
plt.grid()
plt.annotate('maximum',
             xy = (kopt, np.mean(bic,axis=1)[kopt-2]),      
             xytext = (kopt-6, np.mean(bic,axis=1)[kopt-2]-2000),
             weight = 'bold', 
             color = 'xkcd:coral',
             arrowprops = {
                 'arrowstyle': '->',
                 'connectionstyle': 'arc3',
                 'color': 'xkcd:coral',
                 'alpha': 0.7
             })

plt.annotate('K=%d'%kopt,
             xy = (kopt, np.mean(bic,axis=1)[0]),     # click: 0, purchase: 48 
             xytext = (kopt-6, np.mean(bic,axis=1)[0]), # click: 0, purchase: 48 
             weight = 'bold', 
             color = 'xkcd:coral',
             arrowprops = {
                 'arrowstyle': '->',
                 'connectionstyle': 'arc3',
                 'color': 'xkcd:coral',
                 'alpha': 0.7
             })
plt.title("Click Embedding of Following Group",fontsize=16,fontweight='bold')

"""
plt.title("Click Embedding of Ignoring Group",fontsize=16,fontweight='bold')
plt.title("Purchase Embedding of Following Group",fontsize=16,fontweight='bold')
plt.title("Purchase Embedding of Ignoring Group",fontsize=16,fontweight='bold')

"""
plt.show()