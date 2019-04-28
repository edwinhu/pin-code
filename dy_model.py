# -*- coding: utf-8 -*-

# numpy for matrix algebra
import pandas as pd
from pandas import isnull
import numpy as np
from numpy import log, exp

from scipy.misc import logsumexp
from scipy.linalg import inv
import scipy.optimize as op
from itertools import imap, izip_longest

from sklearn.cluster import k_means

# import common functions
from common import *

class DYModel(object):

    def __init__(self,a,d,tn,te,es,eb,us,ub,ss,sb,n=1,t=252):
        """
        Initializes parameters of a Duarte and Young (2009) Model
        
        a : $\alpha$, the unconditional probability of an information event
        d : $\delta$, the unconditional probability of good news
        tn: $\theta_n$, the unconditional probability of a symmetric order-flow shock when there is news
        te: $\theta_e$, the unconditional probability of a symmetric order-flow shock when there is no news
        es : $\epsilon_s$, the average number of sells on a day with no news
        eb : $\epsilon_b$, the average number of buys on a day with no news
        us : $\mu_s$, the average number of (additional) sells on a day with news
        ub : $\mu_b$, the average number of (additional) buys on a day with news
        ss : $\Delta_s$, the average number of (additional) sells on a day with SOS
        sb : $\Delta_b$, the average number of (additional) buys on a day with SOS

        n : the number of stocks to simulate, default 1
        t : the number of periods to simulate, default 252 (one trading year)

        id	News	SOS
        ====================
        0	None	No
        1	None	Yes
        2	Bad	No
        3	Bad	Yes
        4	Good	No
        5	Good	Yes
        """
        
        # Assign model parameters
        # Flatten params
        a, d, tn, te, es, eb, us, ub, ss, sb, n, t \
        = [x.item() if isinstance(x,np.ndarray) else x for x in (a, d, tn, te, es, eb, us, ub, ss, sb, n, t)]
        self.a, self.d, self.tn, self.te, self.es, self.eb, self.us, self.ub, self.ss, self.sb, self.N, self.T \
        = a, d, tn, te, es, eb, us, ub, ss, sb, n, t
        self.states, self.buys, self.sells, self.alpha, self.alpha_g, self.alpha_b = [[]]*6

        # Represent tree in terms of a pandas DataFrame
        index = pd.MultiIndex.from_product([['None', 'Bad', 'Good'],['No', 'Yes']],names=['News','SOS'])
        tree = pd.DataFrame({
            'event': [1-a]*2+[a]*4,
            'news': [1]*2+[1-d]*2+[d]*2,
            'sos': [1-tn,tn]+[1-te,te]*2,
            'buys': [eb,eb+sb,eb,eb+sb,eb+ub,eb+ub+sb],
            'sells': [es,es+ss,es+us,es+ss+us,es,es+ss]
        },index=index)
        tree['prob'] = tree['event']*tree['news']*tree['sos']
        self.tree = tree
        
        self.states = self._draw_states()
        self.buys = np.random.poisson(tree['buys'].iloc[self.states.flat]).reshape(n,t)
        self.sells = np.random.poisson(tree['sells'].iloc[self.states.flat]).reshape(n,t)
        self.alpha = compute_alpha(a, d, tn, te, eb, es, ub, us, sb, ss, self.buys, self.sells)

    def _draw_states(self):
        """
        id	News	SOS
        ====================
        0	None	No
        1	None	Yes
        2	Bad	No
        3	Bad	Yes
        4	Good	No
        5	Good	Yes
        """
        events = np.random.binomial(1, self.a, (self.N,self.T))
        news = np.random.binomial(1, self.d, (self.N,self.T))
        sos_n = np.random.binomial(1, self.tn, (self.N,self.T))
        sos_e = np.random.binomial(1, self.te, (self.N,self.T))

        states = np.empty((self.N,self.T))
        states[(events == 0) & (sos_n == 0)] = 0
        states[(events == 0) & (sos_n == 1)] = 1
        states[(events == 1) & (news == 0) &  (sos_e == 0)] = 2
        states[(events == 1) & (news == 0) &  (sos_e == 1)] = 3
        states[(events == 1) & (news == 1) & (sos_e == 0)] = 4
        states[(events == 1) & (news == 1) & (sos_e == 1)] = 5

        return states

def _lf(eb, es, n_buys, n_sells):
    return -eb+n_buys*log(eb)-lfact(n_buys)-es+n_sells*log(es)-lfact(n_sells)

def _ll(a, d, tn, te, eb, es, ub, us, sb, ss, n_buys, n_sells):
    return np.array([log((1-a)*(1-tn))+_lf(eb,es,n_buys,n_sells),
                     log((1-a)*(tn))+_lf(eb+sb,es+ss,n_buys,n_sells),
                     log((a)*(1-tn)*(1-d))+_lf(eb,es+us,n_buys,n_sells),
                     log((a)*(tn)*(1-d))+_lf(eb+sb,es+ss+us,n_buys,n_sells),
                     log((a)*(1-tn)*(d))+_lf(eb+ub,es,n_buys,n_sells),
                     log((a)*(tn)*(d))+_lf(eb+sb+ub,es+ss,n_buys,n_sells)])

def compute_alpha(a, d, tn, te, eb, es, ub, us, sb, ss, n_buys, n_sells):
    ll = _ll(a, d, tn, te, eb, es, ub, us, sb, ss, n_buys, n_sells)
    llmax = ll.max(axis=0)
    y = exp(ll-llmax)
    alpha = y[2:].sum(axis=0)/y.sum(axis=0)
    
    return alpha

def loglik(theta, n_buys, n_sells):
    a,d,t,eb,es,ub,us,sb,ss = theta
    ll = _ll(a,d,t,t,eb,es,ub,us,sb,ss,n_buys,n_sells)
    
    return sum(logsumexp(ll,axis=0))

def fit(n_buys, n_sells, starts=10, maxiter=100, 
        a=None, d=None, t=None, eb=None, es=None, 
        ub=None, us=None, sb=None, ss=None,
        se=None, **kwargs):
    nll = lambda *args: -loglik(*args)
    bounds = [(0.00001,0.99999)]*3+[(0.00001,np.inf)]*6
    ranges = [(0.00001,0.99999)]*3
    
    a0,d0,t0 = [x or 0.5 for x in (a,d,t)]
    km = k_means(np.array([n_buys,n_sells]).T,2,random_state=1234)[0]
    sb0,ss0 = [l or r for (l,r) in zip((sb,ss),km[1]-km[0])]
    eb0,es0= [l or r for (l,r) in zip((eb,es),km[0])]
    oib = n_buys - n_sells
    ub0 = ub or np.mean(abs(oib))
    us0 = us or ub0
    
    res_final = [a0,d0,t0,eb0,es0,ub0,us0,sb0,ss0]
    stderr = np.zeros_like(res_final)
    f = nll(res_final,n_buys,n_sells)
    for i in range(starts):
        rc = -1
        j = 0
        while (rc != 0) & (j <= maxiter):
            if (None in (res_final)) or i:
                a0,d0,t0 = [np.random.uniform(l,np.nan_to_num(h)) for (l,h) in ranges]
                eb0,es0,ub0,us0,sb0,ss0 = np.random.poisson([eb,es,ub,us,sb,ss])
            res = op.minimize(nll, [a0,d0,t0,eb0,es0,ub0,us0,sb0,ss0], method=None,
                              bounds=bounds, 
                              args=(n_buys,n_sells))
            rc = res['status']
            check_bounds = list(imap(lambda x,y: x in y, res['x'], bounds))
            if any(check_bounds):
                rc = 3
            j+=1
        if (res['success']) & (res['fun'] <= f):
            f,rc = res['fun'],res['status']
            res_final = res['x'].tolist()
            stderr = 1/np.sqrt(inv(res['hess_inv'].todense()).diagonal())
    param_names = 'a,d,t,eb,es,ub,us,sb,ss'.split(',')        
    output = dict(zip(param_names+['f','rc'],
                    res_final+[f,rc]))
    if se:
        output = {'params': dict(zip(param_names,res_final)),
                  'se': dict(zip(param_names,stderr)),
                  'stats':{'f': res['fun'],'rc': res['status']}
                 } 
    return output

def cpie_mech_dy(turn):
    km = k_means(np.array([turn]).T,2,random_state=1234)[0].flatten()
    km = np.sort(np.append(km,km.mean()))
    mech = np.zeros_like(turn)
    mech[((turn > km[0]) & (turn <= km[1]))] = 1
    mech[(turn > km[2])] = 1
    return mech

if __name__ == '__main__':
    
    import pandas as pd
    from regressions import *

    a,d,t,eb,es,ub,us,sb,ss = [0.489493,0.575609,0.285586,0.285586,219.196989,248.681991,93.444485,73.451801,81.519250,83.443071]
    N = 1000
    T = 252

    model = DYModel(a, d, t, t, es, eb, us, ub, ss, sb, n=N, t=T)
    
    buys = to_series(model.buys)
    sells = to_series(model.sells)

    aoib = abs(buys-sells)
    turn = buys+sells
    alpha = to_series(model.alpha)

    def run_regs(df):
        # run regression
        m = []
        m.append(partial_r2(df['alpha'],df[['aoib','aoib2']], df[['aoib','aoib2','turn','turn2']]))
        out = pd.DataFrame(m, columns=['results'])
        out.index.names = ['model']
        return out

    regtab = pd.DataFrame({'alpha':alpha,'aoib':aoib,'aoib2':aoib**2,'turn':turn,'turn2':turn**2})
    
    res = run_regs(regtab)

    print est_tab(res.results, est=['params','tvalues'], stats=['rsquared','rsquared_sp'])

        
        
        
