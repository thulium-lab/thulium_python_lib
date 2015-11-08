
# coding: utf-8

# In[ ]:

#%pylab inline
from IPython.html.widgets import FloatProgress
from IPython.display import display
from matplotlib.cbook import flatten
from numpy import *
import re
import json


# In[ ]:

class JsonCustomEncoder(json.JSONEncoder):
    """ <cropped for brevity> """
    def default(self, obj):
        if isinstance(obj, (ndarray, number)):
            return obj.tolist()
        elif isinstance(obj, (complex, complex)):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        return json.JSONEncoder.default(self, obj)


# In[ ]:

# removed coeffitioent 2 in gaussian functions in exponents
def gaussian(x,N,x0,sigma, background):
    """Returns value of a 1D-gaussian with the given parameters"""
    #from numpy import sqrt,pi,exp
    return N / (sigma * sqrt(pi)) * exp(-(x - x0)**2/(sigma**2)) + background

def gaussian2D(N, x0, y0, sigma_x, sigma_y, background):
    """Returns a 2D-gaussian function with the given parameters"""
    #from numpy import pi,exp
    sigma_x = float(sigma_x)
    sigma_y = float(sigma_y)
    return lambda x,y: N / (sigma_x * sigma_y  * pi) * exp(
                        -(((x - x0) / sigma_x)**2 + ((y - y0) / sigma_y)**2)) + background


# In[1]:

class Image_Basics():
    """Basic image processing, with only center determination and background substructing. 
        Fits are available and called later.
        isgood parameter is needed for indicating if smth wrong with image during it's processing 
        (i.e. if image is blank)"""
    def __init__(self,image, image_url='not_defined'):
        self.image_url = image_url
        self.image = image
        c_pos = self.image.argmax()
        self.center_pos = array([c_pos//self.image.shape[1], c_pos%self.image.shape[1]])
        self.image = self.bgd_substract()
        self.total = sum(self.image)
        self.isgood = True
    def bgd_substract(self, slice_to_c = (-20,-1)):
        """ Substracts background, which is calculated using vertical strip at right side"""
        data_for_bgd_det = self.image[:,slice_to_c[0]:slice_to_c[1]]
        av = sum(data_for_bgd_det,1)/data_for_bgd_det.shape[1]
        return self.image - tile(av[:,newaxis],self.image.shape[1])
    def fit_gaussian1D(self, axis, width=10):
        """Returns (height, x, width_x,  background)
        the gaussian parameters of a 2D distribution found by a fit"""
        from scipy.optimize import curve_fit
        data = sum(self.image, axis)
        popt, pcov = curve_fit(gaussian, range(len(data)), data, p0=(self.total, argmax(data), width, 0))
        return popt
    def fit_gaussian2D(self):
        """Returns (height, y, x, width_y, width_x, background)
        the gaussian parameters of a 2D distribution found by a fit"""
        from scipy import optimize
        params = (self.total, self.fit1D_y[1], self.fit1D_x[1], self.fit1D_y[2], self.fit1D_x[2], 0)
        errorfunction = lambda p: ravel(gaussian2D(*p)(*indices(self.image.shape)) -
                                 self.image)
        popt, success = optimize.leastsq(errorfunction, params)
        return popt
    # removed coeffitioent 2 in gaussian functions in exponents


# In[ ]:

class Load_Image():
    """ Class for loading image. 
        Parameters are clear, dview - view on remote engins (to run in parallel)
        To change filter function from default (gaussian_filter), do instance.filter_functions = new_function, it should take
        only 1 parameter
        """
    def __init__(self, dview = None, do_fit1D_x=True, do_fit1D_y=True, do_fit2D=False, do_filtering=False):
        from scipy.ndimage import gaussian_filter#, median_filter
        self.do_fit1D_x = do_fit1D_x
        self.do_fit1D_y = do_fit1D_y
        self.do_fit2D = do_fit2D
        self.do_filtering = do_filtering
        self.filter_function = gaussian_filter
        self.filter_param = 1 # for gaussian filtering
        dview['loader'] = self # loader - name of instance to load images, send this class to subengins for parallel
    def load_image(self,image_url):
        """ Loads individual image and performs fits. If fit wasn't found, isgood flag is set to False
            If do_filtering is True, filters image
            Depending on flags, adds follow attribute:
            fit1Dx = [total, x0, sigma_x, background]
            fit1Dy = [total, y0, sigma_y, background]
            fit2D  = [total, y0, x0, sigma_y, sigma_x, background]
        """
        from matplotlib.pyplot import imread
        from re import findall
        data = Image_Basics(imread(image_url),image_url)
        (data.folderN, data.shotN, data.shot_typeN) = map(float, findall(r"[-+]?\d*\.\d+|\d+", image_url)[-3:])
        if self.do_filtering:
            data.image = self.filter_function(data.image,self.filter_param)
        try: 
            if self.do_fit1D_x:
                data.fit1D_x = data.fit_gaussian1D(0)
            if self.do_fit1D_y:
                data.fit1D_y = data.fit_gaussian1D(1)
            if self.do_fit2D:
                data.fit2D = data.fit_gaussian2D()
        except RuntimeError:
            print("RuntimeError, couldn't find fit for image", data.image_url)
            data.isgood = False
        return data
    def __call__(self,directory,lview = None):
        """ Construct list of image_urls and load them in parallel. Bad images are not filtered out here, only indication
            by isgood flag
        """
        import os, re
        dirs = [os.path.join(directory,dr) for dr in os.listdir(directory) if re.match(r'[-+]?[0-9.]+ms',dr)]
        files_to_load = [os.path.join(dr,fl) for dr in dirs for fl in os.listdir(dr) if re.match(r'.*_\d+.png',fl)]
        res = lview.map(lambda x: self.load_image(x), files_to_load)
        res.wait_interactive()
        #all_data = list(flatten(res.result))
        #all_data = list(flatten(map(self.load_image, files_to_load)))
        print(''.join([x['stdout'] for x in res.metadata]))
        print('Total number of images: ', len(res.result))
        return res.result


# In[ ]:

def rearrange_data(all_data,sift_by_isgood=True):
    """ Rearranges data from flat list to dictionary simultaneously dropping bad images if flag sift_by_isgood is set
    to True (it can be set to False when signaks are weak)
    """
    dataD = dict()
#     w = FloatProgress(min=0, max=len(all_data),value=0)
#     w.description='Rearranging in progress...'
#     display(w)
    for elem in all_data:
#         w.value +=1
        if sift_by_isgood and (not elem.isgood):
            print('Not good ',elem.image_url)
            continue
        if elem.folderN not in dataD:
            dataD[elem.folderN] = dict()
        d = dataD[elem.folderN]
        if elem.shot_typeN not in d:
            d[elem.shot_typeN] = []
        d[elem.shot_typeN].append(elem)
#     w.bar_style='success'
#     w.description = 'Rearranging Done'
    print('Rearranging to dictionary is complited')
    return dataD


# In[ ]:

class Avr_Image():
    """ Class for average data, has all attributes as Load_Image
        """
    def __init__(self, dview=None, do_fit1D_x=True, do_fit1D_y=True, do_fit2D=True, do_filtering=False, do_sifting=True):
        from scipy.ndimage import gaussian_filter#, median_filter
        self.do_fit1D_x = do_fit1D_x
        self.do_fit1D_y = do_fit1D_y
        self.do_fit2D = do_fit2D
        self.do_sifting = do_sifting
        self.do_filtering = do_filtering
        self.filter_function = gaussian_filter
        self.filter_param = 1 # for gaussian filtering
        dview['averager']=self
    def check_image(self,avr_image, image): 
        # check is based on comparison position of center of average image and individual image
        conf_int = 0.1
        x = image.isgood             and abs((avr_image.center_pos[0] - image.center_pos[0]) / avr_image.center_pos[0]) < conf_int             and abs((avr_image.center_pos[1] - image.center_pos[1]) / avr_image.center_pos[1]) < conf_int
        return x
    def avr_image(self,folderN, shot_typeN, image_list):
        """ For average image there are (if) exist everaged data from each image and its standatd deviation:
        data.total_mean, data.total_std, fit1D_x(y)_mean, fit1D_x(y)_std, data.fit2D_mean, data.fit2D_std
        If any fit wasn't found avr_image is not added to the dictionary"""
        data = Image_Basics(mean([d.image for d in image_list],0), "folder=%f,shot_typeN=%i"%(folderN,shot_typeN))
        # sifting 
        if self.do_sifting:
            # construct new image list with valid images
            new_image_list = [image for image in image_list if self.check_image(data, image)]
            image_list = new_image_list
            # construct new average image from this new list
            data = Image_Basics(mean([d.image for d in image_list],0), "folder=%f,shot_typeN=%i"%(folderN,shot_typeN))
        try: 
            if self.do_fit1D_x:
                data.fit1D_x = data.fit_gaussian1D(0)
            if self.do_fit1D_y:
                data.fit1D_y = data.fit_gaussian1D(1)
            if self.do_fit2D:
                data.fit2D = data.fit_gaussian2D()
        except RuntimeError:
            print("RuntimeError, couldn't find fit for image", data.image_url)
            data.isgood = False
            return []
        data.total_mean = mean([d.total for d in image_list],0)
        data.total_std = std([d.total for d in image_list],0)
        if not all([x.isgood for x in image_list]):
            print('There are bad images in ',"folder=%f,shot_typeN=%i"%(folderN,shot_typeN))
        else:
            if hasattr(image_list[0],'fit1D_x'):
                data.fit1D_x_mean = mean([d.fit1D_x for d in image_list],0)
                data.fit1D_x_std = std([d.fit1D_x for d in image_list],0)
            if hasattr(image_list[0],'fit1D_y'):
                data.fit1D_y_mean = mean([d.fit1D_y for d in image_list],0)
                data.fit1D_y_std = std([d.fit1D_y for d in image_list],0)
            if hasattr(image_list[0],'fit2D'):
                data.fit2D_mean = mean([d.fit2D for d in image_list],0)
                data.fit2D_std = std([d.fit2D for d in image_list],0)
        return (folderN, shot_typeN, data)  
    def __call__(self, dataD, lview):
        """ Construct average image
        """
        images_to_avr = []
        for folderN, folder_dict in dataD.items():
            for shot_typeN, image_list in folder_dict.items():
                images_to_avr.append((folderN,shot_typeN,image_list))
        res = lview.map(lambda *x: self.avr_image(*x), *zip(*images_to_avr))
        res.wait_interactive()
        print(''.join([x['stdout'] for x in res.metadata]))
        avr_data_list = res.result
        avr_data_dict = dict()
        for elem in avr_data_list:
            if len(elem) != 0:
                avr_data_dict[elem[0]] = avr_data_dict.get(elem[0],dict())
                avr_data_dict[elem[0]][elem[1]]=elem[2]
        return avr_data_dict


# In[ ]:

def mod_avrData(avr_dataD, folderN_calib,n_atom_calib, lin_dim_calib):
    """ modifys avr_dataD dictionary with
        1 Calibrating x-value(foldeN) in respect with measured (i.e. changes it to Gs or MHz value)
        2 Drops from data image attribute (image itself is not needed for futher analizis)
        4 Changes numbers of atoms in fits and totals and linear dimentions (line x0 and width) from pixels to teal
            size in μm
    """
    def mod_val(key, value):
        # modifys data on Natoms and raal_size
        val = copy(value)
        if key.startswith("total"):
            return n_atom_calib(val)
        elif key.startswith("center_pos"):
            return lin_dim_calib(val)
        elif key.startswith("fit"):
            val[0] = n_atom_calib(val[0])
            val[-1] = n_atom_calib(val[-1])
            val[1:-1] = lin_dim_calib(val[1:-1])
            return val
        else:
            return val
        
        
    navrD = dict()
    for key in sorted(avr_dataD.keys()):
        new_key = folderN_calib(key)
        navrD[new_key] = dict()
        for shotTypeN, avr_im in avr_dataD[key].items():
            navrD[new_key][shotTypeN] = {key: mod_val(key,value) for (key, value) in avr_im.__dict__.items() if key != 'image'}
    return navrD


# In[ ]:

def get_avr_data_for_plot(avr_dataD, shot_typeN, norm_func, attribute, index=None): 
    """ Construct data for plot from avr_dataD dictionary
        Automaticaly sorts keys and drops a key if there no avr_image for specified shot_typeN
        Automaticaly normalizes 'y' values using norm_func, and constructs standard deviation error.
        If this is not needed just do after d_plot['yerr'] = None"""
    d_plot = dict()
    ks = avr_dataD.keys()
    ks_f = []
    for k in sorted(ks):
        if shot_typeN in avr_dataD[k].keys():
            ks_f.append(k)
        else:
            print('avr_dataD has no average image for folderN=%i shot_typeN=%i' % (k,shot_typeN))
    d_plot['x'] = array(ks_f)
    d_plot['y'] = norm_func(array([get_value(avr_dataD[xx][shot_typeN],attribute,index) for xx in d_plot['x']]))
    try:
        d_plot['yerr'] = norm_func(array([get_value(avr_dataD[xx][shot_typeN],attribute +'_std',index) for xx in d_plot['x']]))
    except AttributeError:
        d_plot['yerr']=None
    return d_plot


# In[ ]:

def get_value(obj, attribute, index):
    """retruns obj.attibute[index] or obj.attribute if index is not defined"""
    if index != None:
        return getattr(obj,attribute)[index]
    else:
        return getattr(obj,attribute)


# In[ ]:

def constract_data(dictionary, shot_typeN, attribute, index = None):
    """The most usefull tool. Returns x_data and y_data list already suitable for plotting (i.e. with the same length)
    dictionary - the dictionary to extract data from (i.e. dataD or avr_dataD)
    shot_typeN - type of the shot sequence (the last number in image name)
    attribute - which attribute of data instance to use !!!look at help for Avr_inf and Image_Image and all their 
    parents
    index - if attribute is a list, specifies which paticular data to use"""
    x_data = array([])
    y_data = array([])
    import collections
    for folderN, f_dict in dictionary.items():
        if f_dict == {}:
            continue
        if isinstance(f_dict[shot_typeN], collections.Iterable):
            temp_arr = [get_value(elem, attribute, index) for elem in f_dict[shot_typeN]]
        else:
            temp_arr = [get_value(f_dict[shot_typeN], attribute, index)]
        y_data = append(y_data, temp_arr)
        x_data = append(x_data, ones(len(temp_arr)) * folderN)
    return x_data, y_data


# In[ ]:

def sift(dataD, confidence_interval = 0.1):
    """Sifts (filters) data on empty images by using average information and comperes centers  of 1D gaussian fits.
    If difference is larger the 'confidence_interval' from the average value, the image would be removed from dataD"""
    w = FloatProgress(min=0, max = len(dataD), value=0)
    w.description='Sifting in progress...'
    display(w)
    for folderN, folder_dict in dataD.items():
        w.value += 1
        for shot_typeN, shot_list in folder_dict.items():
            #print(folderN, shot_typeN)
            avr_inf = Avr_inf(shot_list, do_fit2D=False)
            to_remove = []
            for elem in shot_list:
                if abs(elem.fit1D_x[1]-avr_inf.fit1D_x[1])/avr_inf.fit1D_x[1] > confidence_interval or                     abs(elem.fit1D_y[1]-avr_inf.fit1D_y[1])/avr_inf.fit1D_y[1] > confidence_interval:
                        to_remove.append(elem)
            for elem in to_remove:
                print('remove element',shot_list.index(elem), elem.image_url )
                shot_list.remove(elem)
    w.bar_style='success'
    w.description = 'Sifting Done'


# In[ ]:

def single_directory_sift(d_tuple, confidence_interval):
    """Function to use in parallel sigting
    !!! works slower than without parallelism"""
    folderN, folder_dict = d_tuple
    temp_dict = dict()
    for shot_typeN, shot_list in folder_dict.items():
        #print(folderN, shot_typeN)
        avr_inf = Avr_inf(shot_list, do_fit2D=False)
        to_remove = []
        for elem in shot_list:
            if abs(elem.x_data_fit[1]-avr_inf.x_data_fit[1])/avr_inf.x_data_fit[1] > confidence_interval or                 abs(elem.y_data_fit[1]-avr_inf.y_data_fit[1])/avr_inf.y_data_fit[1] > confidence_interval:
                    to_remove.append(elem)
        for elem in to_remove:
            print('remove element',shot_list.index(elem), elem.image_url )
            shot_list.remove(elem)
    return folderN, folder_dict 


# In[ ]:

def normalise_avr_image(dictionary, signal_shot, calibration_shot, attribute, index=None, do_fit2D = True):
    """normalize image from evarage dictionary using attribute[index] value - usually 'total' or 'x_data_fit[0]'
        returns constracted dictionary (like what returns 'average_data()' function"""
    norm_data = dict()
    w = FloatProgress(min=0, max=len(dictionary),value=0)
    w.description='Normalizing in progress...'
    display(w)
    for folderN, f_dict in dictionary.items():
        w.value += 1
        norm_data[folderN] = dict()
        norm_data[folderN][signal_shot] = Image_Fitted(f_dict[signal_shot].image / 
                                          get_value(f_dict[calibration_shot],attribute,index),do_fit2D)
    w.bar_style='success'
    w.description = 'Normalizing Done'
    print('Normalization is complited')
    return norm_data


# In[ ]:

def normalise_individual_image(dictionary, signal_shot, calibration_shot, attribute, index=None, do_fit2D = False):
    """normalize each image using attribute[index] value - usually 'total' or 'x_data_fit[0]'
        returns constracted dictionary (like what returns 'load_data()' function"""
    norm_data = dict()
    w = FloatProgress(min=0, max=len(dictionary),value=0)
    w.description='Normalizing in progress...'
    display(w)
    for folderN, f_dict in dictionary.items():
        w.value += 1
        calibrated_images = []
        for s_elem in f_dict[signal_shot]:
            c_elems = [c_elem for c_elem in f_dict[calibration_shot] if c_elem.shotN == s_elem.shotN]
            if c_elems == []:
                print('s_elem.image_url has no calibration image')
                continue
            calibrated_images = append(calibrated_images, 
                                       Image_Fitted(s_elem.image / get_value(c_elems[0],attribute,index), do_fit2D))
        if calibrated_images != []:
            norm_data[folderN] = dict()
            norm_data[folderN][signal_shot] = calibrated_images
    w.bar_style='success'
    w.description = 'Normalizing Done'
    print('Normalization is complited')
    return norm_data


# In[ ]:

class N_atoms:
  """
  Natoms = N_atoms(gain, exp, power, width, delta) - создать объект класса
  Natoms(signal) - считает число атомов. по сути в этом месте просто умножение на число
  signal - параметр фита
  [exposure]=us
  [power]=mW
  [width]=mm
  [delta]=MHz
  [gamma]=MHz
  [angle]=1
  """
  def __init__(self, gain=100, exposure=300, power=2.7, width=2.27, delta = 0, gamma = 10, angle = 1./225, Isat = 0.18, hw = 6.6*3/0.41*10**(-11)):
    self.s = 2*power/3.141592654/width**2/Isat
    self.rho = self.s/2/(1+self.s+(2*delta/gamma)**2)
    self.p = 9.69*0.001/100/exposure/2.718281828**(3.85/1000*gain)/gamma/hw/angle/self.rho
  
  def __call__(self, signal):
    return signal*self.p


# In[ ]:

def real_size(x, binning=2, pixel_size = 22.3/4):
    return x * binning * pixel_size


# In[ ]:

def drop_data(data_lists, points):
    """ Drop points from all lists in data_list, mask is constracted using first list in data_lists"""
    mask = array([not(x in points) for x in data_lists[0]])
    res = []
    for data_list in data_lists:
        list.append(res,data_list[mask])
    return res


# In[ ]:

def drop_by_number(data, *numbers):
    """Drops point by its serial number"""
    for k in data.keys():
        if data[k]!=None and type(data[k])!=str:
            data[k] = delete(data[k],numbers)


# In[1]:

def drop_by_x(data, *points):
    """Drops point by its 'x' value"""
    mask = array([not(x in points) for x in data['x']])
    for k in data.keys():
        if data[k]!=None and type(data[k])!=str:
            data[k] = data[k][mask]


# In[ ]:

def data2_sort(x,y):
    """ Sort both array x and y using x-array as criteria"""
    res = array(sorted(zip(x,y), key=lambda x: x[0]))
    return res[:,0],res[:,1]


# In[ ]:

x_lbl_default = 'time, ms'
y_lbl_default = 'N atoms'

def parametric_resonance(conf_params):
    if 'XAXIS' in conf_params:
        xaxis = re.findall('([0-9.]+)(\w+)',conf_params['XAXIS'])[0]
        x_lbl = xaxis[1] 
        return x_lbl, y_lbl_default,lambda y: y * float(xaxis[0])
    else:
        x_lbl = 'kHz'
        return x_lbl, y_lbl_default, lambda y: y

def feshbach(conf_params):
    x_lbl = 'magnetic field, Gs'
    if 'CONF' not in conf_params: conf_params['CONF'] = 'BH'
    if 'OFFSET' not in conf_params: conf_params['OFFSET'] = '0'
    return x_lbl, y_lbl_default, lambda x: FB_conf[conf_params['CONF'].upper()][0] * x +                                         FB_conf[conf_params['CONF'].upper()][1] * float(conf_params['OFFSET'])
    
def temperature(conf_params):
    y_lbl = 'cloud size, μm'
    return x_lbl_default,y_lbl, lambda y: y

def clock(conf_params):
    x_lbl = 'AOM frequency, MHz'
    return x_lbl, y_lbl_default, lambda y: y

def as_measurement(conf_params,dirs,folder):
    as_folder = [x for x in dirs if x.startswith(re.findall('\A\d+\s+\w+\s+(\d+)', folder)[0])]
    if len(as_folder) == 0:
        return x_lbl_default, y_lbl_default, lambda x: x
    else:
        return get_x_calibration(as_folder[0], dirs)

def lifetime(conf_params):
    return x_lbl_default, y_lbl_default, lambda y: y
meas_types = dict()

FB_conf = {'BH':(1,0.1)}
FB_conf['SH']=(FB_conf['BH'][1],FB_conf['BH'][0])

# 'FB' means that magnetic field is scanned
meas_types['FB'] = feshbach
meas_types['T']  = temperature
# lambda function is used up to now due to axis are default ones
meas_types['LT'] = lifetime
meas_types['CL'] = clock
meas_types['PR'] = parametric_resonance
meas_types['AS'] = as_measurement


# In[ ]:

def get_x_calibration(folder,dirs):
    meas_type = re.findall('\d+\s+(\w+)',folder)
    conf = re.findall('(\w+)=(\S+)+', folder)
    conf_params = {key.upper(): value for (key, value) in conf}
    if len(meas_type) == 0 or meas_type[0].upper() not in meas_types:
        # no calibration for x axis and labels are default ms and Natoms
        return x_lbl_default, y_lbl_default, lambda x: x
    elif meas_type[0].upper() == 'AS':
        return as_measurement(conf_params,dirs,folder)
    else:
        return meas_types[meas_type[0].upper()](conf_params)


# In[ ]:

print('Done importing, module image_processing')

