import os
import _pickle as cPickle
import pandas as pd
import glob, dill
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from copy import deepcopy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from utils import draw_network, animate_network

import matplotlib.cm as cm

method = 'cluster'

basedir = 'results'

# dclasses = [[['var','acf-high','uncoupled'],['var','acf-low','uncoupled']],]
dclasses = [[['var','acf-high','uncoupled'],['var','acf-low','uncoupled']],]
dclasses = [[['linear'],['nonlinear']],]

sclasses = [['basic'],
            ['distance'],
            ['causal'],
            ['infotheory'],
            ['spectral'],
            ['wavelet'],
            ['misc']]

if method == 'cluster':
    focal_stats = [
                ('dtw_constraint-itakura',0.6),
                ('psi_wavelet_max_fs-1_fmin-0_fmax-0-5_max',0.4),
                ('icoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5',0.4),
                ('ccm_E-None_mean',0.6),
                ('coint_johansen_max_eig_stat_order-0_ardiff-10',0.75),
                ('mi_gaussian',0.6),
                ('te_kraskov_NN-4_k-max-10_tau-max-4',0.6),
                ('gc_gaussian_k-max-10_tau-max-2',0.7)
                ]
else:
    focal_stats = ['cov_EmpiricalCovariance',
                    'dcorr',
                    'phi_star_t-1_norm-0',
                    'dtw_constraint-itakura',
                    'psi_wavelet_max_fs-1_fmin-0_fmax-0-5_max',
                    'icoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5',
                    'ccm_E-None_mean',
                    'coint_johansen_max_eig_stat_order-0_ardiff-10',
                    'te_kraskov_NN-4_k-max-10_tau-max-4',
                    'gc_gaussian_k-max-10_tau-max-2']

num_in_net = 20

use_sns = True

basedir = os.path.dirname(os.path.abspath(__file__))
savedir = os.path.join(basedir,'plots','comparison')

path = os.path.join(basedir,'results','library_df.pkl')
print(f'Loading CorrelationFrame from {path}...')
with open(path,'rb') as f:
    cf = cPickle.load(f)

def process_adj(A):
    keep_stats = ['-sq' not in idx and 'ctw' not in idx and 'xme' not in idx for idx in A.index]
    return A.loc[keep_stats,keep_stats]

ss_adj = cf.get_average_correlation(absolute=True,remove_insig=False,thresh=0.8)
ss_adj = process_adj(ss_adj)
statnames = ss_adj.columns

y = 1-ss_adj.fillna(0).values[np.triu_indices(ss_adj.shape[0],1)]
Z = linkage(y,metric='euclidean',method='weighted',optimal_ordering=True)

for i, dtypes in enumerate(dclasses):

    cf.set_dgroups(dtypes)

    mdfs = {}
    ss_adjs = {}
    for d, dtype in enumerate(dtypes):

        dstr = '_'.join(dtype)

        dtype_ids = [_i for _i, col in enumerate(cf.mdf.index.droplevel([1,2])) if cf.get_dgroup_ids([col])[0] == d]
        mdf = cf.mdf.iloc[dtype_ids]

        ss_adj = mdf.abs().groupby('Source statistic').mean()
        thresh = 0.8
        ss_adj = ss_adj.dropna(thresh=ss_adj.shape[0]*thresh,axis=0).dropna(thresh=ss_adj.shape[1]*thresh,axis=1).sort_index(axis=1)
        ss_adj = process_adj(ss_adj)

        statnames = ss_adj.columns

        mdfs[dstr] = mdf
        ss_adjs[dstr] = ss_adj

        cf.set_sgroups(sclasses)
        groups = pd.Series(cf.get_sgroup_names(statnames))
        lut = dict(zip(groups.unique(),sns.color_palette('pastel', groups.unique().size)))
        colors = {f: g for f, g in zip(statnames,groups.map(lut).values)}

        csavedir = os.path.join(savedir,dstr+'_'+method)
        try:
            os.mkdir(csavedir)
        except FileExistsError:
            pass
        for statpair in focal_stats:
            if method == 'cluster':
                s, cutoff = statpair
                cmodules = fcluster(Z,cutoff,criterion='distance')
                statmodmap = {f: m for f,m in zip(statnames,cmodules)}
                mod = statmodmap[s]
                mystatnames = [f for f in statnames if statmodmap[f] == mod]
            elif method == 'closest':
                s = statpair
                # mystatdf = ss_adj[s].sort_values(ascending=False)[:num_in_net]
                mystatdf = ss_adjs['_'.join(dtypes[0])][s].sort_values(ascending=False)[:num_in_net]
                mystatnames = [s for s in mystatdf.index if s in ss_adj.columns]
            myadj = ss_adj.loc[mystatnames,mystatnames]
            try:
                draw_network(myadj,f=s,squared=False,node_color=colors,color_labels=lut,savedir=csavedir,pos=None,labels_on=True)
            except KeyError:
                pass