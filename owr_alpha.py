import sys
import pandas as pd
import numpy as np
from numpy.linalg import det, inv
from owr_model import compute_alpha, _compute_up_e, _compute_up_n
from numba import jit

owr_data = pd.read_csv("/mnt/data/pin/{0}_OWR.csv".format(sys.argv[1]))
alphas = np.zeros((owr_data.shape[0],3))

@jit
def compute_alphas(data,alphas):
    event_id = None
    for i,row in enumerate(data):
        t = row[0]
        x = row[2:5]
        if row[1] != event_id:
            event_id = row[1]
            aa = row[5]
            
            s_n = row[6:15].reshape(3,3)
            s_e = row[15:24].reshape(3,3)
        
            dsn, isn = det(s_n), inv(s_n)
            dse, ise = det(s_e), inv(s_e)
            dsd = dse/dsn
            isd = isn - ise
        
        #up_n = _compute_up_n(x,aa,dsn,isn)
        #up_e = _compute_up_e(x,aa,dse,ise)
        #alpha = up_e / (up_n + up_e)
        alpha = compute_alpha(x,aa,dsd,isd)
        alphas[i] = np.array([t,event_id,alpha])

compute_alphas(owr_data.values,alphas)

as_df = pd.DataFrame(alphas,columns=['t','event_id','alpha'])
as_df.to_csv("/mnt/data/pin/{0}_owr_alpha.csv".format(sys.argv[1]))
