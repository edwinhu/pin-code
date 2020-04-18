import ipyparallel as ipp
import pandas as pd
import tables as tb
import os
import argparse

parser = argparse.ArgumentParser(description='Model and year to estimate.')
parser.add_argument('model', type=str, nargs='?', default='gpin')
args = parser.parse_args()
print(vars(args))

rc = ipp.Client(cluster_id="{0}-{1}".format(args.model,'fix'))
print(len(rc))
dv = rc[:]
dv.push(vars(args))
lv = rc.load_balanced_view()

h5 = tb.open_file('/scratch/nyu/hue/taqdf_1319.h5', mode='r')
df = h5.get_node('/data/table')
e = pd.read_csv('/home/nyu/eddyhu/{0}-1319.csv'.format(args.model))
idx = e.loc[e.rc!=0 | pd.isnull(e.f)]
idx = list(zip(idx.permno, idx.yyyy))

@ipp.interactive
def est(x):
    import os
    import pandas as pd
    import tables as tb
    import json
    os.chdir('/home/nyu/eddyhu/git/pin-code')

    import eo_model as eo
    import gpin_model as gpin
    import owr_model as owr
    
    d = pd.read_hdf('/scratch/nyu/hue/taqdf_1319.h5',
                    where='permno=={permno} & yyyy=={yyyy}'.format(permno=x[0], yyyy=x[1]))
    d = d.dropna()
    if model == 'eo':
        r = eo.fit(d.n_buys, d.n_sells, starts=5)
    if model == 'gpin':
        r = gpin.fit(d.n_buys, d.n_sells, starts=5)
    if model == 'owr':
        r = owr.fit(d.uy_e, d.ur_d, d.ur_o, starts=5)
    r.update({'permno':int(x[0]),'yyyy':int(x[1])})
    fname = '/home/nyu/eddyhu/pin-estimates/{model}/{permno}-{yyyy}.json'.format(model=model, permno=x[0], yyyy=x[1])
    with open(fname, 'w') as outfile:
        json.dump(r, outfile)
    return r

res = lv.map_async(est, idx)
res.wait()
