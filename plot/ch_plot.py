import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import json
import pickle

# Click embedding in Following group
st=19
ed=29
with open("../pickles/pos_click_ch_50_k%dto%d.pickle"%(st,ed),'rb') as file:
    ch_dict=pickle.load(file)

"""
# Click embedding in Ignoring group
st=15
ed=25
with open("../pickles/neg_click_ch_50_k%dto%d.pickle"%(st,ed),'rb') as file:
    ch_dict=pickle.load(file)


# Purchase embedding in Following group
st=6
ed=16
with open("../pickles/pos_purchase_ch_50_k%dto%d.pickle"%(st,ed),'rb') as file:
    ch_dict=pickle.load(file)


# Purchase embedding in Ignoring group
st=4
ed=14
with open("../pickles/neg_purchase_ch_50_k%dto%d.pickle"%(st,ed),'rb') as file:
    ch_dict=pickle.load(file)

"""

ch_first=ch_dict['first']
ch_last=ch_dict['last']

ch_first=ch_first.reshape(11,1,-1)# 20*50
ch_last=ch_last.reshape(11,1,-1) # 20*50


ch_score=np.concatenate((ch_first,ch_last),axis=1) # 19*2*10
box_score=ch_score.reshape(22,-1)
box_score=box_score.tolist()

fig = plt.figure(figsize=(8,6))  
ax = plt.subplot() 
outlier = dict(markerfacecolor='b',alpha=0.4)
positions=[[x]*2 for x in range(st,ed+1)]
positions=[x for sublist in positions for x in sublist]

medianprops  = dict(color='orange')
box=ax.boxplot(box_score,positions=positions,flierprops=outlier,patch_artist=True,medianprops=medianprops)
for idx,patch in enumerate(box['boxes']):
	if idx%2==0:
		patch.set_facecolor('magenta')
	else:
		patch.set_facecolor('yellow')
ax.set_xticks([x for x in range(st,ed+1)])

l=[x for x in range(st,ed+1)]
ax.set_xticklabels([str(x) for x in l])

plt.grid(axis='y')
plt.xlabel('Number of Clusters',fontsize=16)
plt.ylabel('Calinski-Harabasz Score',fontsize=16)
plt.ylim((0, 100))
patch0 = mpatches.Patch(color='magenta', label='First')
patch1 = mpatches.Patch(color='yellow', label='Last')

plt.legend(handles=[patch0,patch1])
plt.title("Click Embedding of Following Group",fontsize=16,fontweight='bold')

"""
plt.title("Click Embedding of Ignoring Group",fontsize=16, fontweight='bold')
plt.title("Purchase Embedding of Following Group",fontsize=16, fontweight='bold')
plt.title("Purchase Embedding of Ignoring Group",fontsize=16, fontweight='bold')

"""

plt.show()