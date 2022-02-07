from pynats.calculator import CorrelationFrame
import os
import _pickle as cPickle
import dill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

path = os.path.join('results', 'dd_adj.csv')
dd_adj = pd.read_csv(path,index_col=0)
mvtsnames = dd_adj.columns

np.fill_diagonal(dd_adj.values,np.nan)

focal_mvts = ['oscillator_sync_k--1_conn_all-M5-T500',
                'wave-1D_M-9_T-1000',
                'chaotic_brownian_motion_of_defect_alpha-1-85_epsilon-0-1_M10_T100',
                'hcp_rsfMRI_S2_R-113-122',
                'spatiotemporal_intermittency_i_alpha-1-7522_epsilon-0-00115_M20_T500']

n_insets = 10
offset = 0.05
xlen = 0.08
ylen = 0.15
for mvts in focal_mvts:
    corrs = dd_adj[mvts]
    feature_values = feature_matrix[dd_adj[mvts].index]
    rhos = feature_values.T.corrwith(corrs,method='spearman').dropna().sort_values()
    print(rhos[-30:])

    topfeature = rhos.index[-1]
    topfeature_values = feature_values.T[topfeature]
    fig, ax = plt.subplots()
    ax.plot(corrs,topfeature_values,'k.',zorder=0)
    ax.set_xlabel(f'Correlation to {mvts}')
    ax.set_ylabel(' vs. '.join(topfeature[:-1]))

    # Include insets
    valid_ids = ~topfeature_values.isna() & ~corrs.isna()
    # rsort = corrs[valid_ids].sort_values()
    # fvsort = topfeature_values[rsort.index]
    fvsort = topfeature_values[valid_ids].sort_values()
    rsort = corrs[fvsort.index]
    inset_ids = np.linspace(0,len(fvsort)-1,n_insets,dtype=int)
    for i in inset_ids:
        name = fvsort.index[i]
        Z = database[name]['data'].T
        cx = rsort[name]+0.01
        cy = fvsort[name]
        Y, X = np.mgrid[cy:cy+ylen:ylen/Z.shape[0],cx:cx+xlen:xlen/Z.shape[1]]
        ax.pcolormesh(X,Y,Z,cmap=sns.color_palette('icefire_r', as_cmap=True),zorder=1)

    plt.show()
