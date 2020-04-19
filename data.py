import os
import pandas as pd
from importlib import reload
os.chdir('/home/nyu/eddyhu/git/pin-code')
import eo_model as eo
import gpin_model as gpin
import owr_model as owr

# setup data
df = pd.read_sas('/scratch/nyu/hue/taqdfx_all6.sas7bdat')
df['yyyy'] = df.yyyy.astype('int')
df['date'] = df.DATE
df['permno'] = df.permno.astype('int')
df['ticker'] = df.symbol_15.str.decode('UTF-8')
df.set_index('permno yyyy'.split(),inplace=True)
c = df.groupby(level=(0,1))\
    ['n_buys n_sells ur_d ur_o uy_e'.split()]\
    .count().min(axis=1)
c.name = 'count_min'
df1 = df.join(c)
df1.loc[df1.count_min>=230]\
    ['date ticker n_buys n_sells ur_d ur_o uy_e'.split()]\
    .to_hdf('/scratch/nyu/hue/taqdf_1319.h5','data',format='table')

d = pd.read_hdf('/scratch/nyu/hue/taqdf_1319.h5',where='permno==11850 & yyyy==2015')

# rest run of each model
eo.fit(d.n_buys,d.n_sells,starts=1)
gpin.fit(d.n_buys,d.n_sells,starts=1)
owr.fit(d.uy_e,d.ur_d,d.ur_o,starts=1)