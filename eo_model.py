# -*- coding: utf-8 -*-

# numpy for matrix algebra
import numpy as np
from numpy import log, exp

from scipy.misc import logsumexp
from scipy.linalg import inv
import scipy.optimize as op

# import common functions
from common import *

class EOModel(object):

    def __init__(self,a,d,es,eb,u,n=1,t=252):
        """Initializes parameters of an Easley and O'Hara Sequential Trade Model
        
        a : $\alpha$, the unconditional probability of an information event
        d : $\delta$, the unconditional probability of good news
        es : $\epsilon_s$, the average number of sells on a day with no news
        eb : $\epsilon_b$, the average number of buys on a day with no news
        u : $\mu$, the average number of (additional) trades on a day with news

        n : the number of stocks to simulate, default 1
        t : the number of periods to simulate, default 252 (one trading year)
        """

        # Assign model parameters
        self.a, self.d, self.es, self.eb, self.u, self.N, self.T = a, d, es, eb, u, n, t
        self.states = self._draw_states()
        self.buys = np.random.poisson((eb+(self.states == 1)*u))
        self.sells = np.random.poisson((es+(self.states == -1)*u))
        self.alpha = compute_alpha(a, d, eb, es, u, self.buys, self.sells)

    def _draw_states(self):
        """Draws the states for N stocks and T periods.

        In the Easley and O'Hara sequential trade model at the beginning of each period nature determines whether there is an information event with probability $\alpha$ (a). If there is information, nature determines whether the signal is good news with probability $\delta$ (d) or bad news $1-\delta$ (1-d).

        A quick way to implement this is to draw all of the event states at once as an `NxT` matrix from a binomial distribution with $p=\alpha$, and independently draw all of the news states as an `NxT` matrix from a binomial with $p=\delta$. 
        
        An information event occurs for stock i on day t if `events[i][t]=1`, and zero otherwise. The news is good if `news[i][t]=1` and bad if `news[i][t]=-1`. 

        The element-wise product of `events` with `news` gives a complete description of the states for the sequential trade model, where the state variable can take the values (-1,0,1) for bad news, no news, and good news respectively.

        self : EOSequentialTradeModel instance which contains parameter definitions
        """
        events = np.random.binomial(1, self.a, (self.N,self.T))
        news = np.random.binomial(1, self.d, (self.N,self.T))
        news[news == 0] = -1

        states = events*news

        return states

def _lf(eb, es, n_buys, n_sells):
    return -eb+n_buys*log(eb)-lfact(n_buys)-es+n_sells*log(es)-lfact(n_sells)

def _ll(a, d, eb, es, u, n_buys, n_sells):
    return np.array([log(a*(1-d))+_lf(eb,es+u,n_buys,n_sells), 
                   log(a*d)+_lf(eb+u,es,n_buys,n_sells), 
                   log(1-a)+_lf(eb,es,n_buys,n_sells)])
            
def compute_alpha(a, d, eb, es, u, n_buys, n_sells):
    '''Compute the conditional alpha given parameters, buys, and sells.

    '''
    ll = _ll(a, d, eb, es, u, n_buys, n_sells)    
    llmax = ll.max(axis=0)
    y = exp(ll-llmax)
    alpha = y[:-1].sum(axis=0)/y.sum(axis=0)
    
    return alpha

def loglik(theta, n_buys, n_sells):
    a,d,eb,es,u = theta
    ll = _ll(a, d, eb, es, u, n_buys, n_sells)
    
    return sum(logsumexp(ll,axis=0))
            
def fit(n_buys, n_sells, starts=10, maxiter=100, 
        a=None, d=None, eb=None, es=None, u=None,
        se=None, **kwargs):

    nll = lambda *args: -loglik(*args)
    bounds = [(0.00001,0.99999)]*2+[(0.00001,np.inf)]*3
    ranges = [(0.00001,0.99999)]*2
    
    a0,d0 = [x or 0.5 for x in (a,d)]
    eb0,es0 = eb or np.mean(n_buys), es or np.mean(n_sells)
    oib = n_buys - n_sells
    u0 = u or np.mean(abs(oib))

    res_final = [a0,d0,eb0,es0,u0]
    stderr = np.zeros_like(res_final)
    f = nll(res_final,n_buys,n_sells)
    for i in range(starts):
        rc = -1
        j = 0
        while (rc != 0) & (j <= maxiter):
            if (None in (res_final)) or i:
                a0,d0 = [np.random.uniform(l,np.nan_to_num(h)) for (l,h) in ranges]
                eb0,es0,u0 = np.random.poisson([eb,es,u])
            res = op.minimize(nll, [a0,d0,eb0,es0,u0], method=None,
                              bounds=bounds, args=(n_buys,n_sells))
            rc = res['status']
            check_bounds = list(imap(lambda x,y: x in y, res['x'], bounds))
            if any(check_bounds):
                rc = 3
            j+=1
        if (res['success']) & (res['fun'] <= f):
            f,rc = res['fun'],res['status']
            res_final = res['x'].tolist()
            stderr = 1/np.sqrt(inv(res['hess_inv'].todense()).diagonal())
    param_names = ['a','d','eb','es','u']
    output = dict(zip(param_names+['f','rc'],
                    res_final+[f,rc]))
    if se:
        output = {'params': dict(zip(param_names,res_final)),
                  'se': dict(zip(param_names,stderr)),
                  'stats':{'f': f,'rc': rc}
                 }
    return output

def cpie_mech(turn):
    mech = np.zeros_like(turn)
    mech[turn > turn.mean()] = 1
    return mech

if __name__ == '__main__':
    
    import pandas as pd
    from regressions import *

    a = 0.41
    d = 0.58
    es = 2719
    eb = 2672
    u = 2700

    N = 1000
    T = 252

    model = EOModel(a,d,es,eb,u,n=N,t=T)

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

    print(est_tab(res.results, est=['params','tvalues'], stats=['rsquared','rsquared_sp']))
