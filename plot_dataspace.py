from pynats.calculator import CorrelationFrame
import os
import _pickle as cPickle
import utils
import matplotlib.pyplot as plt

import seaborn as sns

reducer = 'tsne'

datadir = 'results'
path = os.path.join(datadir, 'library_df.pkl')
print(f'Loading CorrelationFrame from {path}...')
with open(path,'rb') as f:
    cf = cPickle.load(f)
print('Done.')

utils.dataspace(cf,classes=[['real'],
                            ['synthetic']
                            ],
                            reducer=reducer,plot_nas=False,include_size=False)

utils.dataspace(cf,classes=[['real','fmri'],
                            ['EEG'],
                            ['MEG'],
                            ['river'],
                            ['stocks'],
                            ['forex'],
                            ['earthquake'],
                            ['epidemic','incidence'],
                            ['epidemic','cumulative'],
                            ],
                            reducer=reducer,plot_nas=False,include_size=False)
plt.tight_layout()

utils.dataspace(cf,classes=[['vma'],
                            ['var'],
                            ['noise'],
                            ['partial differential equation'],
                            ['coupled ordinary differential equation'],
                            ['stochastic differential equation'],
                            ['synthetic','fmri'],
                            ['logistic map'],
                            ['oscillator'],
                            ],reducer=reducer,plot_nas=False,
                            include_size=False)
plt.tight_layout()


utils.dataspace(cf,classes=[['real','fmri'],
                            ['EEG'],
                            ['MEG'],
                            ['river'],
                            ['stocks'],
                            ['forex'],
                            ['earthquake'],
                            ['epidemic','incidence'],
                            ['epidemic','cumulative'],
                            ['vma'],
                            ['var'],
                            ['noise'],
                            ['partial differential equation'],
                            ['coupled ordinary differential equation'],
                            ['stochastic differential equation'],
                            ['synthetic','fmri'],
                            ['logistic map'],
                            ['oscillator'],
                            ],reducer=reducer,plot_nas=False,
                            include_size=False)
plt.tight_layout()

utils.dd_cluster(cf,classes=[['real','fmri'],
                            ['EEG'],
                            ['MEG'],
                            ['river'],
                            ['stocks'],
                            ['forex'],
                            ['earthquake'],
                            ['epidemic','incidence'],
                            ['epidemic','cumulative'],
                            ])

plt.show()