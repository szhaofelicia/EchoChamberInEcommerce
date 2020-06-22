from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import pickle
from scipy import stats

from tqdm import tqdm
import json

np.random.seed(0)

"""
Following group
"""
with open('../jsons/pos_user_display_diversity.json','r') as load_f:
    user_display = json.load(load_f)
pos_div=np.array(list(user_display.values())) 
pos_samp, _ = train_test_split(pos_div, test_size=0.6531,random_state=0) # display

fir_div=pos_samp[:,0] 
last_div=pos_samp[:,1] 

plt.grid()
plt.hist(fir_div,50,color='xkcd:turquoise', edgecolor='tab:gray')
plt.xlabel('Content Diversity',fontsize=16)
plt.ylabel('Number of Users',fontsize=16)
plt.title('First Blocks of Foloowing Group',fontsize=16,fontweight='bold')
plt.xticks(np.arange(0.950,1.155,0.025))
plt.yticks(np.arange(0,455,50))
plt.show()

plt.grid()
plt.hist(last_div,50,color='xkcd:turquoise', edgecolor='tab:gray')
plt.xlabel('Content diversity',fontsize=16)
plt.ylabel('Number of Users',fontsize=16)
plt.title('Last Blocks of Following Group',fontsize=16,fontweight='bold')
plt.xticks(np.arange(0.950,1.155,0.025))
plt.yticks(np.arange(0,455,50))
plt.show()


"""
Ignoring group
"""
with open('../jsons/neg_user_display_diversity.json','r') as load_f:
    user_display = json.load(load_f)
neg_div=np.array(list(user_display.values())) 

fir_div=neg_div[:,0] 
last_div=neg_div[:,1] 

plt.grid()
plt.hist(fir_div,50,color='xkcd:turquoise', edgecolor='tab:gray')
plt.xlabel('Content Diversity',fontsize=16)
plt.ylabel('Number of Users',fontsize=16)
plt.title('First Blocks of Ignoring Group',fontsize=16,fontweight='bold')
plt.xticks(np.arange(0.950,1.155,0.025))
plt.yticks(np.arange(0,455,50))
plt.show()

plt.grid()
plt.hist(last_div,50,color='xkcd:turquoise', edgecolor='tab:gray')
plt.xlabel('Content diversity',fontsize=16)
plt.ylabel('Number of Users',fontsize=16)
plt.title('Last Blocks of Ignoring Group',fontsize=16,fontweight='bold')
plt.xticks(np.arange(0.950,1.155,0.025))
plt.yticks(np.arange(0,455,50))
plt.show()
