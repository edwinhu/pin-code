# -*- coding: utf-8 -*-

# numpy for matrix algebra
from pandas import isnull
import numpy as np
from numpy import log, exp

from scipy.special import gamma
from scipy.misc import logsumexp
import scipy.optimize as op

# import common functions
from common import *

from numba import jit

class EEOWModel(object):

    def __init__(self,G,Omega,Phi,Gamma,alpha,delta,Psi0=[0,0],n=1,t=252,t_burn=None):
        """Initializes parameters of EPIN model

        n : the number of stocks to simulate, default 1
        t : the number of periods to simulate, default 252 (one trading year)
        """

        # Assign model parameters
        self.G, self.Omega, self.Phi, self.Gamma, self.alpha, self.delta, self.N, self.T = G,Omega,Phi,Gamma,alpha,delta,n,t
        if not t_burn:
            t_burn=t/10
        states = self._draw_states(alpha,delta,t+t_burn,n)
        buys = np.zeros_like(states)
        sells = np.zeros_like(states)
        Psi = np.zeros((t+t_burn,2,n))
        Psi[0] = np.tile(Psi0,n).reshape((2,n))
        self.Psi = self.sim_Psi(n,alpha,G,Omega,Phi,Gamma,Psi,buys,sells,states)[t_burn:]
        self.mu = self.Psi[:,0]/alpha
        self.epsilon = self.Psi[:,1]/2
        self.buys = buys[t_burn:]
        self.sells = sells[t_burn:]
        self.states = states[t_burn:]
        self.cpie = compute_cpie(alpha, delta, self.epsilon, self.epsilon, self.mu, self.buys, self.sells)
    
    def _draw_states(self,alpha,delta,t,n):
        """Draws the states for N stocks and T periods.

            In the Easley and O'Hara sequential trade model at the beginning of each period nature determines whether there is an information event with probability $\alpha$ (a). If there is information, nature determines whether the signal is good news with probability $\delta$ (d) or bad news $1-\delta$ (1-d).

            A quick way to implement this is to draw all of the event states at once as an `NxT` matrix from a binomial distribution with $p=\alpha$, and independently draw all of the news states as an `NxT` matrix from a binomial with $p=\delta$. 

            An information event occurs for stock i on day t if `events[i][t]=1`, and zero otherwise. The news is good if `news[i][t]=1` and bad if `news[i][t]=-1`. 

            The element-wise product of `events` with `news` gives a complete description of the states for the sequential trade model, where the state variable can take the values (-1,0,1) for bad news, no news, and good news respectively.

            self : EOSequentialTradeModel instance which contains parameter definitions
            """
        events = np.random.binomial(1, alpha, (t,n))
        news = np.random.binomial(1, delta, (t,n))
        news[news == 0] = -1
        states = events*news
        return states

    @jit
    def sim_Psi(self,n,alpha,G,Omega,Phi,Gamma,Psi,buys,sells,states):
        for t in range(1,len(Psi)):
            epsilon = Psi[t-1][1]/2
            mu = Psi[t-1][0]/alpha
            buys[t] = np.random.poisson((epsilon+(states[t-1] == 1)*mu))
            sells[t] = np.random.poisson((epsilon+(states[t-1] == -1)*mu))
            aoib = abs(buys[t]-sells[t])
            bo = (buys[t]+sells[t])-aoib
            Z = np.array([aoib,bo])
            Psi[t] = _compute_Psi(t,np.tile(G,n),np.tile(Omega,n),Phi,Gamma,Z,Psi[t-1])
            #Psi[t] = (np.tile(Omega*np.exp(G*t),N) + np.dot(Phi,Psi[t-1])*np.tile(np.exp(G),N) + np.dot(Gamma,Z))
        return Psi

def _compute_Psi(t,G,Omega,Phi,Gamma,Z,Psitm1):
    Psi = (Omega*np.exp(G*t) + np.dot(Phi,Psitm1)*np.exp(G) + np.dot(Gamma,Z))
    Psi = (Psi > 0)*Psi + (Psi <= 0)*np.ones_like(Psi)
    return Psi    

@jit
def compute_Psi(G,Omega,Phi,Gamma,Z,Psi):
    for t in range(1,len(Z)):
        Psi[t] = _compute_Psi(t,G,Omega,Phi,Gamma,Z[t],Psi[t-1])
        #(Omega*np.exp(G.T*t) + np.dot(Phi,Psi[t-1])*np.exp(G.T) + np.dot(Gamma,Z[t]))
    return Psi

def _lf(eb, es, n_buys, n_sells):
    return -eb+n_buys*log(eb)-lfact(n_buys)-es+n_sells*log(es)-lfact(n_sells)

def _ll(a, d, eb, es, u, n_buys, n_sells):
    return np.array([log(a*(1-d))+_lf(eb,es+u,n_buys,n_sells), 
                   log(a*d)+_lf(eb+u,es,n_buys,n_sells), 
                   log(1-a)+_lf(eb,es,n_buys,n_sells)])
            
def compute_cpie(a, d, eb, es, u, n_buys, n_sells):
    '''Compute the conditional alpha given parameters, buys, and sells.

    '''
    ll = _ll(a, d, eb, es, u, n_buys, n_sells)    
    llmax = ll.max(axis=0)
    y = exp(ll-llmax)
    cpie = y[:-1].sum(axis=0)/y.sum(axis=0)
    return cpie

# loglik and fit are copied from EPIN model, needs to be modified
def loglik(theta, n_buys, n_sells):
    return NotImplementedError
    a,d,eb,es,u = theta
    ll = _ll(a, d, eb, es, u, n_buys, n_sells)
    
    return sum(logsumexp(ll,axis=0))
            
def fit(n_buys, n_sells, starts=10, maxiter=100, 
        g=np.array([[0],[0]]), 
        Omega=np.array([[0],[0]]),
        Phi=np.array([[0,0],[0,0]]),
        Gamma=np.array([[0,0],[0,0]]),
        alpha=0,delta=0):
    return NotImplementedError
    turn = n_buys + n_sells
    aoib = abs(n_buys - n_sells)
    bo = turn - aoib
    Z = np.vstack((aoib,bo)).T
    Psi = np.zeros_like(Z)
    Psi[0] = Z[0]
    Phi = Phi-Gamma
    Psi = _compute_Psi(Omega,Phi,Gamma,Z,Psi)
    
    # likelihood = likelihood + (-log(sigma(i)^2) - uSq(i)/sigma(i)^2);
    nll = lambda *args: -np.sum(np.log(Psi**2)-Z/Psi**2)
    
    # estimate negative binomial parameters first
    nll = lambda *args: -nbm_ll(*args)
    bounds = [(1,np.inf),(0.000001,0.99999)]
    ranges = [(1,999),(0.000001,0.99999)]

    r0,p0 = r,p
    a0,eta0,d0,th0 = a,eta,d,th
    
    f = np.inf
    for i in range(starts):
        rc = -1
        j = 0
        while (rc != 0) & (j <= maxiter):
            # if any missing or not first iteration try random starts
            if (None in (r0,p0)) or i:
                r0, p0 = [np.random.uniform(l,np.nan_to_num(h)) for (l,h) in ranges]
            res = op.minimize(nll, [r0, p0], method=None,
                              bounds=bounds, args=(turn))
            rc = res['status']
            j+=1
            if (res['success']) & (res['fun'] <= f):
                f = res['fun']
                r,p = res['x']
                s = p/(1-p)

    # estimate rest of parameters
    nll = lambda *args: -loglik(*args)
    bounds = [(0.00001,0.99999),(0.00001,np.inf),(0.00001,0.99999),(0.00001,1.0)]
    ranges = [(0.00001,0.99999)]*4

    f = np.inf
    for i in range(starts):
        rc = -1
        j = 0
        while (rc != 0) & (j <= maxiter):
            # if any missing or not first iteration try random starts
            if (None in (a0,eta0,d0,th0)) or i:
                a0,eta0,d0,th0 = [np.random.uniform(l,np.nan_to_num(h)) for (l,h) in ranges]
            res = op.minimize(nll, [a0,eta0,d0,th0], method=None,
                              bounds=bounds, args=(r,p,n_buys,n_sells))
            rc = res['status']
            check_bounds = list(imap(lambda x,y: x in y, res['x'], bounds))
            if any(check_bounds):
                rc = 3
            j+=1
        if (res['success']) & (res['fun'] <= f):
            f = res['fun']
            res_final = res
            a,eta,d,th = res['x']
    return dict(zip(['a','r','p','eta','d','th','f','rc'],
                    [a,r,p,eta,d,th,f,rc]))

if __name__ == '__main__':
    
    import pandas as pd
    from regressions import *

    # estimates from ASH in EEOW paper
    g = np.array([[0.0072],[0.0093]])
    Omega = np.array([[2.1190],[7.8509]])
    Phi = np.array([[0.5204,0.0348],[-1.7298,1.1219]])
    Gamma = np.array([[0.0768,0.0720],[0.3022,0.3316]])
    alpha = 0.4092
    delta = 0.5511
    Psi0 = np.exp([0.921,2.721])
    N = 2
    T = 252

    model = EEOWModel(g,Omega,Phi-Gamma,Gamma,alpha,delta,n=N,t=T)

    buys = to_series(model.buys)
    sells = to_series(model.sells)
    
    aoib = abs(buys-sells)
    turn = buys+sells-aoib
    cpie = to_series(model.cpie)

    def run_regs(df):
        # run regression
        m = []
        m.append(partial_r2(df['cpie'],df[['aoib','aoib2']], df[['aoib','aoib2','turn','turn2']]))
        out = pd.DataFrame(m, columns=['results'])
        out.index.names = ['model']
        return out

    regtab = pd.DataFrame({'cpie':cpie,'aoib':aoib,'aoib2':aoib**2,'turn':turn,'turn2':turn**2})
    
    res = run_regs(regtab)

    print(est_tab(res.results, est=['params','tvalues'], stats=['rsquared','rsquared_sp']))
