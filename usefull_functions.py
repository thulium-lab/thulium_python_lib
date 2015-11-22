
# coding: utf-8

# In[ ]:

def exp_decay(t, N0, tau, background):
    return N0 * exp(- t / tau) + background
def exp_decay_no_bg(t, N0, tau):
    return N0 * exp(- t / tau)
def cloud_expansion(t, T, r0, t0):
    k_b = 1.38e-23
    m = 169 * 1.66e-27
    return sqrt(r0**2 + 2 * k_b * T * (t + 1*t0)**2 / m)
def exp_grouth(t, N0, tau, background):
    return N0 * ( 1 - exp( - t / tau)) + 0*background

def construct_fit_description(fit_func, popt_T):
    """constructs a set of string of type 'variable=value\n' for all [1:] function variables"""
    from inspect import getargspec
    res = ''
    for item in zip(getargspec(fit_func)[0][1:], popt_T):
        params = item[1] if hasattr(item[1],'__iter__') else [item[1]]
        res += str(item[0]) + ' =   ' + '\t'.join(['%.2f'%(x) for x in params]) + '\n'
    res = res.rstrip('\n')
    return res

