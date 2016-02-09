
# coding: utf-8

# In[ ]:

"""
Module for usefull functions
"""


# In[ ]:

import numpy as np

def exp_decay(t, N0, tau, background):
    r'N0 * np.exp(- t / tau) + background'
    return N0 * np.exp(- t / tau) + background

def exp_decay_no_bg(t, N0, tau):
    r'N0 * np.exp(- t / tau)'
    return N0 * np.exp(- t / tau)

def cloud_expansion(t, T, r0, t0):
    r'np.sqrt(r0**2 + 2 * k_b * T * (t + 1*t0)**2 / m)'
    k_b = 1.38e-23
    m = 169 * 1.66e-27
    return np.sqrt(r0**2 + 2 * k_b * T * (t + 1*t0)**2 / m)

def cloud_expansion0(t, T, r0):
    r'cloud_expansion(t, T, r0, 0)'
    return cloud_expansion(t, T, r0, 0)

def exp_grouth(t, N0, tau, background):
    return N0 * ( 1 - np.exp( - t / tau)) + 0*background

def construct_fit_description(fit_func, popt_T):
    """constructs a set of string of type 'variable=value\n' for all [1:] function variables"""
    from inspect import getargspec
    res = ''
    for item in zip(getargspec(fit_func)[0][1:], popt_T):
        params = item[1] if hasattr(item[1],'__iter__') else [item[1]]
        res += str(item[0]) + ' =   ' + '\t'.join(['%.2f'%(x) for x in params]) + '\n'
    res = res.rstrip('\n')
    return res

def lorentz(x, N, x0, sigma, background):
    return N/pi * 1/2 * sigma / ( (x - x0)**2 + (1/2*sigma)**2) + background

# ### Including some losses 

# In[ ]:

def tow_body_loss(t, N0, betta, background):
    r'return 1 / ( betta * t + 1 / N0) + background'
    return 1 / ( betta * t + 1 / N0) + background
def exp_plus_tw_body_decay(t, N0, tau, betta,  background):
    r'return N0 * exp(- t / tau) / ( 1 + betta * N0 * tau * (1 - exp( -t / tau))) + 0 * background'
    return N0 * exp(- t / tau) / ( 1 + betta * N0 * tau * (1 - exp( -t / tau))) + 0 * background
def two_frac_decay(t, N0, N1, tau, betta,  background):
    r'return exp_decay(t, N0, tau, 0) + exp_plus_tw_body_decay(t,N1, tau, betta,  0) + abs(background)'
    return exp_decay(t, N0, tau, 0) + exp_plus_tw_body_decay(t,N1, tau, betta,  0) + abs(background)
def two_frac_decay_no_bg(t, N0, N1, tau, betta, background):
    r'return two_frac_decay(t, N0, N1, tau, betta,0)'
    return two_frac_decay(t, N0, N1, tau, betta,0)

