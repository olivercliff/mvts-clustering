from pynats.calculator import CorrelationFrame
import os
import _pickle as cPickle
import dill

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster, linkage, cophenet, dendrogram, set_link_color_palette, leaders
import matplotlib as mpl

import numpy as np
from scipy.stats import zscore
import random

from utils import plot_clusters, draw_network

random.seed(1)

basedir = os.path.dirname(os.path.abspath(__file__))
savedir = os.path.join(basedir,'plots')

path = os.path.join('results', 'library_df.pkl')
print(f'Loading CorrelationFrame from {path}...')
with open(path,'rb') as f:
    cf = cPickle.load(f)
print('Done.')

print('Getting feature matrix...')
feature_matrix = cf.get_feature_matrix()
print('Done.')

with open('database.pkl','rb') as f:
    database = dill.load(f)

# Normalize all entries
for name in database:
    database[name]['data'] = zscore(database[name]['data'])

path = os.path.join('results', 'dd_adj.csv')
dd_adj = pd.read_csv(path,index_col=0)
mvtsnames = dd_adj.columns

np.fill_diagonal(dd_adj.values,np.nan)

print(f'Average: {dd_adj.mean().mean()}')

method = 'weighted'

y = 1 - dd_adj.fillna(0).values[np.triu_indices(dd_adj.shape[0],1)]
Z = linkage(y,metric='euclidean',method=method,optimal_ordering=True)

threshold = 0.76

C = cophenet(Z,y)
print(f'Cophenetic distance: {C[0]}')   

focal_mvts = [
                ('oscillator_hysteresis_ow--4_nw--1_is-rand_io-neg-M10-T1000',0.5),
                ('spatiotemporal_intermittency_i_alpha-1-7522_epsilon-0-00115_M20_T500',0.4),
                ('chaotic_brownian_motion_of_defect_alpha-1-85_epsilon-0-1_M10_T100',0.3),
                ('wave-1D_M-9_T-1000',0.4),
                ('hcp_rsfMRI_S2_R-113-122',0.5),
                ('hcp_tfMRI_S6_R-310-319',0.47),
                ('mousefMRI_S-0_R-23-24',0.35),
                ]

for mvts, cutoff in focal_mvts:
    cmodules = fcluster(Z,cutoff,criterion='distance')
    mvtsmodmap = {d:m for d,m in zip(mvtsnames,cmodules)}
    print(f'Computing similarity network for {mvts}')
    try:
        mod = mvtsmodmap[mvts]
        nearest_neighbours = [m for m in mvtsnames if mvtsmodmap[m] == mod]
    except KeyError:
        print(f'Time series {mvts} not in matrix.')
        continue
    myadj = dd_adj.loc[nearest_neighbours,nearest_neighbours]

    ts = (0.8,0.4)
    draw_network(myadj,f=mvts,mvts=database,squared=True,savedir=None,seed=1,pos=None,labels_on=True,ts=ts,ws=(2,0.2,0.05))

plt.show()