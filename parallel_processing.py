
# coding: utf-8

# In[ ]:

from IPython import parallel
from matplotlib.cbook import flatten
import thulium_python_lib.image_processing as imp

# Start claster and load_balance_view
rc = parallel.Client()
rc.ids
lview = rc.load_balanced_view()

with rc[:].sync_imports():
    import sys, os
    
get_ipython().magic("px if r'D:\\\\!Data' not in sys.path: sys.path.append(r'D:\\\\!Data')")
get_ipython().magic('px import thulium_python_lib.image_processing as imp')


# In[7]:

def load_data_parallel(directory, do_fit2D = False, do_filtering=False):
    import os, re
    dirs = [os.path.join(directory,dr) for dr in os.listdir(directory) if re.match(r'[-+]?[0-9.]+ms',dr)]
    #from operator import concat
    #from functools import reduce
    res = lview.map(imp.single_directory_load, dirs ,[do_fit2D]*len(dirs), [do_filtering]*len(dirs))
    res.wait_interactive()
    all_data = list(flatten(res.result))
    # print outputs in each kernel
    print(''.join([x['stdout'] for x in res.metadata]))
    print('Total number of images: ', len(all_data))
    return all_data


# In[ ]:

def average_data_parallel(dataD, do_fit2D=True):
    res = lview.map(imp.single_directory_average,dataD.items(),[do_fit2D]*len(dataD))
    res.wait_interactive()
    # print outputs in each kernel
    print(''.join([x['stdout'] for x in res.metadata]))
    return dict(res.result)


# In[ ]:

def sift_parallel(dataD, confidence_interval = 0.1):
    """Currentlu not in use due to slower performance then usual one"""
    res = lview.map(imp.single_directory_average,dataD.items(),[confidence_interval]*len(dataD))
    res.wait_interactive()
    # print outputs in each kernel
    print(''.join([x['stdout'] for x in res.metadata]))
    dataD = dict(res.result)

