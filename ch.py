import numpy as np
import pandas as pd

from random import shuffle,sample

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_rand_score


import pickle
from scipy import stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
import json

np.random.seed(0)

with open('jsons/pos_user_click_embed.json','r') as load_f:
    item_embed_dict = json.load(load_f)
pos_emb=np.array(list(item_embed_dict.values())) # len * 2 * 128