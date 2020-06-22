import numpy as np

from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import json
import pickle

from tqdm import tqdm

def kmeans_clustering(x,k):
	clustering=KMeans(n_clusters=k,random_state=0).fit(x)
	y=clustering.labels_
	return clustering, y


def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value

    Source:
    -------------------------------------------
    https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)]) #>0
    const_term = 0.5 * m * np.log(N) * (d+1) #>0

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)


kmax=50

"""
Following Group: Click Embedding
"""

with open('jsons/pos_user_click_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)

item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128


bic=np.zeros((kmax-1,50))
for k in tqdm(range(2,kmax+1)):
    for t in range(50):
        x_samp, _ = train_test_split(item_emb, test_size=0.512,random_state=t) #click:0.512 # purchase:0.305
        x_train, x_test = train_test_split(x_samp, test_size=0.2,random_state=t)
        fir_list=x_train[:,0,:] # len * 128
        km,clu=kmeans_clustering(fir_list,k)
        bic[k-2][t]=compute_bic(km,fir_list)

bic=np.array(bic)

with open("pickles/pos_click_bic_50.pickle",'wb') as file:
    pickle.dump(bic,file)


"""
Ignoring Group: Click Embedding
"""

with open('jsons/neg_user_click_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)

item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128


bic=np.zeros((kmax-1,50))
for k in tqdm(range(2,kmax+1)):
    for t in range(50):
        x_train, x_test = train_test_split(item_emb, test_size=0.2,random_state=t)
        fir_list=x_train[:,0,:] # len * 128
        km,clu=kmeans_clustering(fir_list,k)
        bic[k-2][t]=compute_bic(km,fir_list)
bic=np.array(bic)


with open("pickles/neg_click_bic_50.pickle",'wb') as file:
    pickle.dump(bic,file)


"""
Following Group: Purchase Embedding
"""

with open('jsons/pos_user_purchase_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)

item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128


bic=np.zeros((kmax-1,50))
for k in tqdm(range(2,kmax+1)):
    for t in range(50):
        x_samp, _ = train_test_split(item_emb, test_size=0.305,random_state=t) #click:0.512 # purchase:0.305
        x_train, x_test = train_test_split(x_samp, test_size=0.2,random_state=t)
        fir_list=x_train[:,0,:] # len * 128
        km,clu=kmeans_clustering(fir_list,k)
        bic[k-2][t]=compute_bic(km,fir_list)

bic=np.array(bic)

with open("pickles/pos_purchase_bic_50.pickle",'wb') as file:
    pickle.dump(bic,file)


"""
Ignoring Group: Purchase Embedding
"""

with open('jsons/neg_user_purchase_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)

item_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128


bic=np.zeros((kmax-1,50))
for k in tqdm(range(2,kmax+1)):
    for t in range(50):
        x_train, x_test = train_test_split(item_emb, test_size=0.2,random_state=t)
        fir_list=x_train[:,0,:] # len * 128
        km,clu=kmeans_clustering(fir_list,k)
        bic[k-2][t]=compute_bic(km,fir_list)
bic=np.array(bic)


with open("../dict_pickles/kselection/neg_purchase_bic_50.pickle",'wb') as file:
    pickle.dump(bic,file)


