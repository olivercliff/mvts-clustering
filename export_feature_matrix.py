import os
import _pickle as cPickle

from utils import plot_clusters, draw_network

basedir = os.path.dirname(os.path.abspath(__file__))
savedir = os.path.join(basedir,'plots')

path = os.path.join('results', 'library_df.pkl')
print(f'Loading CorrelationFrame from {path}...')
with open(path,'rb') as f:
    cf = cPickle.load(f)
print('Done.')

print('Getting feature matrix...')
feature_matrix = cf.get_feature_matrix()

spis = ['ctw' not in s
        and 'lbk' not in s
        and 'gak' not in s
        and 'lmfit_Lasso' not in s
        and ('wavelet' not in s or 'psi' in s) for s in feature_matrix.index]

feature_matrix = feature_matrix.iloc[spis]
feature_matrix = feature_matrix.dropna(axis=0,thresh=0.9*feature_matrix.shape[1]).dropna(axis=1,thresh=0.8*feature_matrix.shape[0])
print(f'Size: {feature_matrix.shape}.')

print('Computing correlations...')
dd_adj = feature_matrix.corr(method='spearman')
fname = os.path.join('results','dd_adj.csv')
print(f'Done, saving to {fname}.')
dd_adj.to_csv(fname)