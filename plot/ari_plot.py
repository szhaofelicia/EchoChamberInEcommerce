import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import json
import pickle

# Click embedding in Following group
with open("../pickles/pos_click_ari_50.pickle",'rb') as file:
    air_dict=pickle.load(file)

st=19
ed=29
"""
# Click embedding in Ignoring group
with open("../pickles/neg_click_ari_50.pickle",'rb') as file:
    air_dict=pickle.load(file)
st=15
ed=25

# Purchase embedding in Following group
with open("../pickles/pos_purchase_ari_50.pickle",'rb') as file:
    air_dict=pickle.load(file)
st=6
ed=16

# Purchase embedding in Ignoring group
with open("../pickles/neg_purchase_ari_50.pickle",'rb') as file:
    air_dict=pickle.load(file)
st=4
ed=14
"""

ari_emb=air_dict['50']

box_score=ari_emb.tolist()

fig = plt.figure(figsize=(8,6))  
ax = plt.subplot() 
outlier = dict(markerfacecolor='b',alpha=0.4)
positions=[x for x in range(st,ed+1)]
# print(positions)

medianprops  = dict(color='yellow')

box=ax.boxplot(box_score,positions=positions,flierprops=outlier,patch_artist=True, medianprops=medianprops)
for idx,patch in enumerate(box['boxes']):
	patch.set_facecolor('xkcd:orangered')
ax.set_xticks([x for x in range(st,ed+1)])

l=[x for x in range(st,ed+1)]
ax.set_xticklabels([str(x) for x in l])

plt.grid(axis='y')
plt.xlabel('Number of Clusters',fontsize=16)
plt.ylabel('Adjusted Rand Index',fontsize=16)
plt.ylim((0.03, 0.15))
patch0 = mpatches.Patch(color='xkcd:orangered', label='ARI')
plt.legend(handles=[patch0])

plt.title("Click Embedding of Following Group",fontsize=16, fontweight='bold')

"""
plt.title("Click Embedding of Ignoring Group",fontsize=16, fontweight='bold')
plt.title("Purchase Embedding of Following Group",fontsize=16, fontweight='bold')
plt.title("Purchase Embedding of Ignoring Group",fontsize=16, fontweight='bold')

"""

plt.show()
