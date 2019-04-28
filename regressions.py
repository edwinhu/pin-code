import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.iolib.summary2 as summ
from statsmodels.iolib.summary2 import summary_params

# for statusbar
import uuid, time
from IPython.display import HTML, Javascript, display

def to_series(a):
    '''Converts a NxT array to NT array'''
    return pd.Series(a.flatten(),index=np.repeat(np.arange(a.shape[0]),a.shape[1]))

def OLS(endog, exog):
    '''Runs OLS with standardized exogenous variables'''
    endog = endog.copy()
    exog = exog.copy()
    endog = (endog - np.mean(endog, axis=0))
    exog = (exog - np.mean(exog, axis=0)) / np.std(exog, axis=0)
    #exog = sm.add_constant(exog)
    res = sm.OLS(endog, exog, missing='drop').fit()

    return res

def partial_r2(endog, exog_sub, exog_full):
    '''Runs OLS and computes partial R2 values'''
    sub = OLS(endog, exog_sub)
    full = OLS(endog, exog_full)

    full.rsquared_sp = full.rsquared - sub.rsquared
    full.rsquared_partial = (full.rsquared - full.rsquared_sp) / (1-full.rsquared_sp)
    full.rsquared_inc = full.rsquared_sp / full.rsquared
    full.rsquared_sub = sub.rsquared

    full.params_sub = sub.params
    full.tvalues_sub = sub.tvalues

    return full
    
def est_tab(res, est=['params','tvalues'], stats=['nobs','rsquared']):
    '''Extracts estimates from results DataFrame.
    
    Params
    ------
    res: `pandas Series` of `statsmodels RegressionResults` objects.
    est: list of estimates (strings)
    stats: list of statistics (strings)

    '''
    est_df = pd.concat([res.apply(lambda x: getattr(x,e)) for e in est], keys=est, axis=1)
    stats_df = pd.concat([res.apply(lambda x: getattr(x,s)) for s in stats], keys=stats, axis=1)
    output = pd.concat([est_df,pd.concat({"stats":stats_df}, axis=1)], axis=1)
    return output

def status(res):
    divid = uuid.uuid4()
    nsim = len(res.metadata)
    prog = res.progress
    remain = np.divide(nsim*1.0,prog)-1
    pb = HTML(
    """
    <div style="border: 1px solid black; width:500px">
      <div id="%s" style="background-color:blue; width:0%%">&nbsp;</div>
    </div> 
    """ % divid)
    js = Javascript("$('div#%s').width('%i%%')" % (divid, prog/nsim*100))
    display(pb)
    display(js)
    print('{0} of {1}, {2:.2f} hours elapsed, {3:.2f} hours left'\
        .format(prog, nsim, res.elapsed/3600, (remain)*(res.elapsed/3600)))

def stdize(x):
    if x.name!='Intercept':
        # if the std deviation is zero or missing, don't rescale it
        # this can happen if cpie_owr_RT is just 1.0 or 0.0
        return (x-np.nanmean(x)).divide(np.nanstd(x) or 1)
    else:
        return x
    
def stars(p, fmt='{:.2f}'):
    est = fmt
    if p < .1:
        est+='*'
    if p < .05:
        est+='*'
    if p < .01:
        est+='*'
    return est    
        
# Vertical summary instance for multiple models
def col_params(result, est_fmt='{:.3f}', sig_fmt='({:.2f})', 
               est_scale = 100.0, sig_scale = 1.0,
               stars=True, use_t=True, xname=[], **kwargs):
    '''Stack coefficients and significance stats in single column
    '''

    # Extract parameters
    res = summary_params(result, xname=xname)
    # Format float
    for col in res.columns[:2]:
        res[col] = res[col].apply(lambda x: est_fmt.format(x*est_scale))
    # tvalues in parentheses
    if use_t:
        res.iloc[:,1] = [sig_fmt.format(x*sig_scale) for x in res.iloc[:,2]]
    else: 
        res.iloc[:,1] = [sig_fmt.format(x*sig_scale) for x in res.iloc[:,3]]
    # Significance stars
    if stars:
        idx = res.iloc[:,3] < .1
        res.iloc[:,0][idx] = res.iloc[:,0][idx] + '*'
        idx = res.iloc[:,3] < .05
        res.iloc[:,0][idx] = res.iloc[:,0][idx] + '*'
        idx = res.iloc[:,3] < .01
        res.iloc[:,0][idx] = res.iloc[:,0][idx] + '*'
    # Stack Coefs and Signif.
    res = res.iloc[:,:2]
    res = res.stack()
    res = pd.DataFrame(res)
    res.columns = [str(result.model.endog_names)]
    return res

# replace the hidden function with our above function
summ._col_params = col_params
