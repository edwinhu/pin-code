# -*- coding: utf-8 -*-
from __future__ import division

# numpy for matrix algebra
import numpy as np
from numpy import log, exp

import pandas as pd

from numpy.linalg import det, inv
from scipy.misc import logsumexp
import scipy.optimize as op
from itertools import imap

# optimization
from numba import jit

class OWRModel(object):
    
    def __init__(self,a,su,sz,si,spd,spo,n=1,t=252):
        """Initializes parameters of an Odders-White and Ready (2008) Sequential Trade Model

        a: $\alpha$, the unconditional probability of an information event
        ... #TODO#
        """

        # Assign model parameters
        self.a, self.su, self.sz, self.si, self.spd, self.spo, self.N, self.T = a, su, sz, si, spd, spo, n, t
        self.s_n, self.s_e = _compute_cov(a, su, sz, si, spd, spo)

        # pre-computing the dets and invs saves a lot of time
        self.dsn, self.isn = det(self.s_n), inv(self.s_n)
        self.dse, self.ise = det(self.s_e), inv(self.s_e)
        self.dsd = self.dsn/self.dse
        self.isd = self.isn - self.ise

        self.states = np.random.binomial(1, a, n*t)
        
        mean = [0]*3
        x_n = np.random.multivariate_normal(mean,self.s_n,n*t).T
        x_e = np.random.multivariate_normal(mean,self.s_e,n*t).T
        self.x = x_n*self.states+x_e+(1-self.states)
        self.oib, self.ret_d, self.ret_o = self.x.reshape(3,n,t)
        self.alpha = _compute_alpha(self.x,
                                    self.a,self.dsd,self.isd)\
                                    .reshape((n,t))

@jit
def _qvmv(x,A):
    """Computes x'Ax.
    """
    m,n = A.shape
    qsum = 0
    
    for i in xrange(m):
        for j in xrange(n):
            qsum += A[i,j] * x[i] * x[j]
            
    return qsum

def _compute_cov(a, su, sz, si, spd, spo):
    # compute covariance matrices
    s_n = [[su**2+sz**2, a**(0.5)*si*su/2, -a**(0.5)*si*su/2],
           [a**(0.5)*si*su/2, spd**2+a*si**2/4, -a*si**2/4],
           [-a**(0.5)*si*su/2, -a*si**2/4, spo**2+(1+a)*si**2/4]]

    s_e = [[(1+1/a)*su**2+sz**2, a**(-0.5)*si*su/2+a**(0.5)*si*su/2, a**(-0.5)*si*su/2 - a**(0.5)*si*su/2],
           [a**(-0.5)*si*su/2+a**(0.5)*si*su/2, spd**2+(1+a)*si**2/4, (1-a)*si**2/4],
           [a**(-0.5)*si*su/2 - a**(0.5)*si*su/2, (1-a)*si**2/4, spo**2+(1+a)*si**2/4]]
    
    return s_n, s_e

def _compute_alpha(x,a,dsd,isd):
    alphas = np.zeros(x.shape[1])
    for i in xrange(x.shape[1]):
        alphas[i] = 1/(1 + (1-a)/a*exp(_lf(x[:,i],dsd,isd)))
    return alphas

def compute_alpha(oib, ret_d, ret_o, a, su, sz, si, spd, spo):
    '''Computes conditional alpha.
    
    Params
    ------
    '''
    if len(a)>1:
        a = a.tolist().pop()
        su = su.tolist().pop()
        sz = sz.tolist().pop()
        si = si.tolist().pop()
        spd = spd.tolist().pop()
        spo = spo.tolist().pop()
    s_n, s_e = _compute_cov(a, su, sz, si, spd, spo)
    dsn, isn = det(s_n), inv(s_n)
    dse, ise = det(s_e), inv(s_e)
    dsd = dsn/dse
    isd = isn-ise
    
    x = np.array([oib,ret_d,ret_o])
    cpie = pd.Series(_compute_alpha(x,a,dsd,isd),index=oib.index)
    return cpie

def _lf(x, det, inv):
    return -0.5*log(det)-0.5*_qvmv(x,inv)

def loglik(theta, oib, ret_d, ret_o):
    a, su, sz, si, spd, spo = theta
    s_n, s_e = _compute_cov(a, su, sz, si, spd, spo)
    dsn, isn = det(s_n), inv(s_n)
    dse, ise = det(s_e), inv(s_e)
    
    x = np.array([oib,ret_d,ret_o])
    t = x.shape[1]
    ll = np.zeros((2,t))
    for i in xrange(t):
        ll[:,i] = log(a)+_lf(x[:,i],dse,ise), log(1-a)+_lf(x[:,i],dsn,isn)
    
    return sum(logsumexp(ll,axis=0))

def fit(oib, ret_d, ret_o, starts=10, maxiter=100, 
        a=None, su=None, sz=None, si=None, spd=None, spo=None,
        se=None, **kwargs):

    nll = lambda *args: -loglik(*args)
    bounds = [(0.00001,0.99999)]+[(0.00001,np.inf)]*5
    ranges = [(0.00001,0.99999)]+[(0.00001,999)]*5
    
    a0,su0,sz0,si0,spd0,spo0 = a or 0.5, su or 0.1, sz or 0.25, si or 0.05, spd or ret_d.std(), spo or ret_o.std()
    res_final = [a0,su0,sz0,si0,spd0,spo0]
    stderr = np.zeros_like(res_final)
    f = nll(res_final,oib,ret_d,ret_o)
    for i in range(starts):
        rc = -1
        j = 0
        while (rc != 0) & (j <= maxiter):
            if (None in (res_final)) or i:
                a0,su0,sz0,si0,spd0,spo0 = [np.random.uniform(l,np.nan_to_num(h)) for (l,h) in ranges]
            res = op.minimize(nll, [a0,su0,sz0,si0,spd0,spo0], method=None,
                              bounds=bounds, args=(oib,ret_d,ret_o))
            rc = res['status']
            check_bounds = list(imap(lambda x,y: x in y, res['x'], bounds))
            if any(check_bounds):
                rc = 3
            j+=1
        if (res['success']) & (res['fun'] <= f):
            f,rc = res['fun'],res['status']
            res_final = res['x'].tolist()
            stderr = 1/np.sqrt(inv(res['hess_inv'].todense()).diagonal())
    param_names = 'a,su,sz,si,spd,spo'.split(',')
    output = dict(zip(param_names+['f','rc'],
                    res_final+[f,rc]))
    if se:
        output = {'params': dict(zip(param_names,res_final)),
                  'se': dict(zip(param_names,stderr)),
                  'stats':{'f': f,'rc': rc}
                 } 
    return output

if __name__ == '__main__':

    import pandas as pd
    import statsmodels.api as sm
    from patsy import dmatrix
    from regressions import *

    a = 0.13796
    su = 0.00132
    sz = 0.00032
    si = 0.01226
    spd = 0.01439
    spo = 0.01159

    T = 252
    N = 1000

    model = OWRModel(a,su,sz,si,spd,spo,n=N,t=T)

    oib = to_series(model.oib)
    ret_d = to_series(model.ret_d)
    ret_o = to_series(model.ret_o)
    alpha = to_series(model.alpha)

    as_df = pd.DataFrame({'oib':oib,'ret_d':ret_d,'ret_o':ret_o,
                          'oib2':oib**2,'ret_d2':ret_d**2,'ret_o2':ret_o**2,
                          'alpha':alpha})

    regtab = dmatrix('oib2 + ret_d2 + ret_o2 + oib:ret_d + oib:ret_o + ret_d:ret_o -1', data=as_df, return_type='dataframe')
    regtab['alpha'] = as_df['alpha']

    regtab_std = (regtab - regtab.apply(np.mean))/regtab.apply(np.std)
    res = sm.OLS(regtab_std['alpha'],regtab_std[['oib2','ret_d2','ret_o2','oib:ret_d','oib:ret_o','ret_d:ret_o']]).fit()
    print(res.summary())
    
