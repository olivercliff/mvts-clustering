from pynats.calculator import CorrelationFrame
import os
import _pickle as cPickle
import utils
import matplotlib.pyplot as plt

import seaborn as sns

reducer = 'pca'

path = os.path.join('results','library_df.pkl')
print(f'Loading CorrelationFrame from {path}...')
with open(path,'rb') as f:
    cf = cPickle.load(f)
print('Done.')

utils.dataspace(cf,classes=[['cauchy-distribution'],
                            ['gamma-distribution'],
                            ['t-distribution'],
                            ['exponential-distribution'],
                            ['normal-distribution','noise'],
                            ['normal-distribution','acf-low'],
                            ['normal-distribution','acf-high'],
                            ],reducer=reducer,plot_nas=False,
                            include_size=False)
plt.tight_layout()

utils.dataspace(cf,classes=[['oscillator','pcnn'],
                            ['oscillator','legion'],
                            ['oscillator','sync'],
                            ['oscillator','hhn'],
                            ['oscillator','fsync'],
                            ],reducer=reducer,plot_nas=False,
                            include_size=False)
plt.tight_layout()

utils.dataspace(cf,classes=[['spatiotemporal_intermittency_ii'],
                            ['chaotic_traveling_wave'],
                            ['chaotic_brownian_motion_of_defect'],
                            ['traveling_wave'],
                            ['spatiotemporal_intermittency_i'],
                            ['pattern_selection'],
                            ['frozen_chaos'],
                            ['defect_turbulence'],
                            ['spatiotemporal_chaos'],
                            ['wave-2D'],
                            ['wave-1D']
                            ],reducer=reducer,plot_nas=False,
                            include_size=False)
plt.tight_layout()

plt.show()

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
                            ['climate'],
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
                            ['climate'],
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
                            ['climate'],
                            ],reducer=reducer,plot_nas=False,
                            include_size=False)
plt.tight_layout()

# utils.dd_cluster(cf,classes=[['real','fmri'],
#                             ['EEG'],
#                             ['MEG'],
#                             ['river'],
#                             ['stocks'],
#                             ['forex'],
#                             ['earthquake'],
#                             ['epidemic','incidence'],
#                             ['epidemic','cumulative'],
#                             ])

plt.show()