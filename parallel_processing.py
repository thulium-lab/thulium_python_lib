
# coding: utf-8

# In[ ]:

from IPython import parallel
rc = parallel.Client()
rc.ids
lview = rc.load_balanced_view()
import thulium_python_lib.image_processing as imp


# In[ ]:

with rc[:].sync_imports():
    import sys, os


# In[ ]:

#sys.path.append(r'D:\\!Data')
get_ipython().magic("px if r'D:\\\\!Data' not in sys.path: sys.path.append(r'D:\\\\!Data')")


# In[ ]:

get_ipython().magic('px import thulium_python_lib.image_processing as imp')


# In[ ]:

def load_data_parallel(directory, do_fit2D = False, do_filtering=False):
    import os, re
    dirs = [os.path.join(directory,dr) for dr in os.listdir(directory) if re.match(r'[-+]?[0-9.]+ms',dr)]
    from operator import concat
    from functools import reduce
    res = lview.map(imp.single_directory_load, dirs ,[do_fit2D]*len(dirs), [do_filtering]*len(dirs))
    res.wait_interactive()
    all_data = reduce(concat,res.result)
    print('Total number of images: ', len(all_data))
    return all_data


# In[ ]:

def average_data_parallel(dataD, do_fit2D=True):
    res = lview.map(imp.single_directory_average,dataD.items(),[do_fit2D]*len(dataD))
    res.wait_interactive()
    return dict(res.result)

