import os
import _pickle as pickler
# import dill as pickler
from scipy.stats import zscore
from pynats.calculator import Calculator
import pandas as pd

datadir = 'results'
datatypes = ['coupled_map_lattice','epidemics','finance','geophysics','neuroscience','oscillators','partial_differential_equations','VARMA','UEA',]
# datatypes = ['coupled_map_lattice']

# Set up calculator for relabelling (used an older version of pynats on the cluster)
calc = Calculator()

names = [s.name for s in calc._statistics]
labels = [s.labels for s in calc._statistics]

for d in datatypes:
    try:
        path = os.path.join(datadir, d + '_df.pkl')
        print(f'Loading CorrelationFrame from {path}...')
        with open(path,'rb') as f:
            cf2 = pickler.load(f)
        print('Done.')

        print('Relabelling statistics...')
        cf2.relabel_statistics(names,labels)
        with open(path,'wb') as f:
            pickler.dump(cf2,f)
        print('Done.')

        path = os.path.join(datadir, d + '_edge.pkl')
        print(f'Loading edges from {path}...')
        with open(path,'rb') as f:
            df2 = pickler.load(f)
        print('Done.')

        print('Merging...')
        try:
            cf.merge(cf2)
            df = df.append(df2)
        except NameError:
            cf = cf2
            df = df2
        print(f'Done.')
    except FileNotFoundError as err:
        print(err)
cf.name = ', '.join(datatypes)

print(f'Included {cf.ddf.shape[1]} calculators.')

# cf.compute_significant_values()

savefile = os.path.join(datadir,'library_df.pkl')
with open(savefile,'wb') as f:
    pickler.dump(cf,f)

savefile = os.path.join(datadir,'library_edges.pkl')
with open(savefile,'wb') as f:
    pickler.dump(df,f)