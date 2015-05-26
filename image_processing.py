
# coding: utf-8

# In[1]:

#%pylab inline
from IPython.html.widgets import FloatProgress
from IPython.display import display
from numpy import *


# In[2]:

def gaussian(x,N,x0,sigma, background):
    """Returns value of a 1D-gaussian with the given parameters"""
    #from numpy import sqrt,pi,exp
    return N / (sigma * sqrt(2 * pi)) * exp(-(x - x0)**2/(2 * sigma**2)) + background

def gaussian2D(N, x0, y0, sigma_x, sigma_y, background):
    """Returns a 2D-gaussian function with the given parameters"""
    #from numpy import pi,exp
    sigma_x = float(sigma_x)
    sigma_y = float(sigma_y)
    return lambda x,y: N / (sigma_x * sigma_y * 2 * pi) * exp(
                        -(((x - x0) / sigma_x)**2 + ((y - y0) / sigma_y)**2) / 2) + background


# In[3]:

class Image_Basics():
    """Basic image processing, with only center determination and background substructing. 
        isgood parameter is needed for indicating if smth wrong with image during it's processing 
        (i.e. if image is blank)"""
    def __init__(self,image):
        if not hasattr(self,'image_url'):
            self.image_url = 'derived'
        self.image = image
        c_pos = self.image.argmax()
        self.center_pos = (c_pos//self.image.shape[1], c_pos%self.image.shape[1])
        self.mimage = self.bgd_substract()
        self.total = sum(self.mimage)
        self.isgood = True
    def bgd_substract(self, slice_to_c = (-20,-1)):
        """ Substracts background, which is calculated using vertical strip at right side"""
        data_for_bgd_det = self.image[:,slice_to_c[0]:slice_to_c[1]]
        av = sum(data_for_bgd_det,1)/data_for_bgd_det.shape[1]
        return self.image - tile(av[:,newaxis],self.image.shape[1])


# In[4]:

class Image_Fitted(Image_Basics):
    """ Ads fitting functionality, namle 1D and 2D gauss fit
        If fitting has failed, it prints error message and delites this image from data
        x_data_fit = [total, x0, sigma_x, background]
        y_data_fit = [total, y0, sigma_y, background]
        fit2D = [total, y0, x0, sigma_y, sigma_x, background]"""
    def __init__(self, image, do_fit2D):
        Image_Basics.__init__(self,image)
        try:
            self.do_fit(do_fit2D)
            self.center_pos = (self.x_data_fit[1], self.y_data_fit[1])
        except RuntimeError:
            print("RuntimeError, couldn't find fit for image", self.image_url)
            self.isgood = False
    def do_fit(self, do_fit2D, width=10):
        """ Does fits"""
        from scipy.optimize import curve_fit
        x_data = sum(self.mimage,0)
        y_data = sum(self.mimage,1)
        popt_x, pcov_x = curve_fit(gaussian, range(len(x_data)), x_data, p0=(self.total, argmax(x_data), width, 0))
        popt_y, pcov_y = curve_fit(gaussian, range(len(y_data)), y_data, p0=(self.total, argmax(y_data), width, 0))
        self.x_data_fit = popt_x
        self.y_data_fit = popt_y
        if do_fit2D:
            self.fit2D = self.fitgaussian2D()
    def fitgaussian2D(self):
        """Returns (height, y, x, width_y, width_x)
        the gaussian parameters of a 2D distribution found by a fit"""
        from scipy import optimize
        params = (self.total, self.y_data_fit[1], self.x_data_fit[1], self.y_data_fit[2], self.x_data_fit[2], 0)
        errorfunction = lambda p: ravel(gaussian2D(*p)(*indices(self.image.shape)) -
                                 self.image)
        p, success = optimize.leastsq(errorfunction, params)
        return p


# In[5]:

class Image_Load(Image_Fitted):
    """ Loads image using relative path, based on Image_Fitted"""
    def __init__(self,image_url, do_fit2D=False):
        from matplotlib.pyplot import imread
        from re import findall
        self.image_url = image_url
        Image_Fitted.__init__(self, imread(image_url), do_fit2D)
        (self.folderN, self.shotN, self.shot_typeN) = map(float, findall(r"[-+]?\d*\.\d+|\d+", self.image_url))


# In[6]:

class Avr_inf(Image_Fitted):
    """ Class for average data, has all attributes as Image_Fitted instance for average image as well as average
        data from each image:
        each_x_data_fit = [total, x0, sigma_x, background]
        each_y_data_fit = [total, y0, sigma_y, background]
        each_fit2D = [total, y0, x0, sigma_y, sigma_x, background] if exists
        """
    def __init__(self,shot_list, do_fit2D=True):
        Image_Fitted.__init__(self,mean([d.mimage for d in shot_list],0), do_fit2D)
        self.each_x_data_fit = mean([d.x_data_fit for d in shot_list],0)
        self.each_y_data_fit = mean([d.y_data_fit for d in shot_list],0)
        if hasattr(shot_list[0],'fit2D'):
            self.each_fit2D = mean([d.fit2D for d in shot_list],0)


# In[7]:

def load_data(do_fit2D = False):
    """Loads all data initially to all_data (unsorted list), and then to dictionary structure dataD
    folderN1  ----    shot_typeN1   ----  [list of Image_Load instances]
                      shot_typeN2   ----  [list of Image_Load instances]
                     ....
    folderN2  ----    shot_typeN1   ----  [list of Image_Load instances]
                      shot_typeN2   ----  [list of Image_Load instances]
                     ....
    By default does not fit each image 2D-gauss"""
    import os, re
    dirs = [dr for dr in os.listdir(os.getcwd()) if re.match(r'[-+]?[0-9.]+ms',dr)]
    all_data = []
    w1 = FloatProgress(min=0, max=len(dirs),value=0)
    w1.description='Loading in progress...'
    display(w1)
    for dr in dirs:
        w1.value += 1
        files = [os.path.join(dr,fl) for fl in os.listdir(dr) if re.match(r'.*_\d+.png',fl)]
        for url in files:
            new_im = Image_Load(url, do_fit2D)
            if new_im.isgood:
                all_data.append(new_im)
    w1.bar_style='success'
    w1.description = 'Loading Done'
    print('Total number of images: ', len(all_data))
    dataD = dict()
    w2 = FloatProgress(min=0, max=len(all_data),value=0)
    w2.description='Rearranging in progress...'
    display(w2)
    for elem in all_data:
        w2.value +=1
        if elem.folderN not in dataD:
            dataD[elem.folderN] = dict()
        d = dataD[elem.folderN]
        if elem.shot_typeN not in d:
            d[elem.shot_typeN] = []
        d[elem.shot_typeN].append(elem)
    w2.bar_style='success'
    w2.description = 'Rearranging Done'
    print('Rearranging to dictionary is complited')
    return dataD


# In[8]:

def average_data(dataD, do_fit2D=True):
    """Averages data from dataD to dictionary structure avr_dataD
    folderN1  ----    shot_typeN1   ----  Avr_inf instances
                      shot_typeN2   ----  Avr_inf instances
                     ....
    folderN2  ----    shot_typeN1   ----  Avr_inf instances
                      shot_typeN2   ----  Avr_inf instances
                     ....
    By default does fit each average image 2D-gauss"""
    avr_dataD = dict()
    w = FloatProgress(min=0, max = len(dataD), value=0)
    w.description='Averaging in progress...'
    display(w)
    for folderN, folder_dict in dataD.items():
        w.value += 1
        avr_dataD[folderN] = dict()
        temp_dict = avr_dataD[folderN]
        for shot_typeN, shot_list in folder_dict.items():
            temp_dict[shot_typeN] = Avr_inf(shot_list, do_fit2D)
    w.bar_style='success'
    w.description = 'Averaging Done'
    print('Averaging is complited')
    return avr_dataD


# In[ ]:

def get_value(obj, attribute, index):
    """retruns obj.attibute[index] or obj.attribute if index is not defined"""
    if index != None:
        return getattr(obj,attribute)[index]
    else:
        return getattr(obj,attribute)


# In[9]:

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
            avr_inf = Avr_inf(shot_list, do_fit2D=False)
            to_remove = []
            for elem in shot_list:
                if abs(elem.x_data_fit[1]-avr_inf.x_data_fit[1])/avr_inf.x_data_fit[1] > confidence_interval and                     abs(elem.y_data_fit[1]-avr_inf.y_data_fit[1])/avr_inf.y_data_fit[1] > confidence_interval:
                        to_remove.append(elem)
            for elem in to_remove:
                print('remove element',shot_list.index(elem), elem.image_url )
                shot_list.remove(elem)
    w.bar_style='success'
    w.description = 'Sifting Done'


# In[ ]:

def normalise_avr_image(dictionary, signal_shot, calibration_shot, attribute, index=None, do_fit2D = True):
    """normalize image from evarage dictionary using attribute[index] value - usually 'total' or 'x_data_fit[0]'
        returns constracted dictionary (like what returns 'average_data()' function"""
    norm_data = dict()
    for folderN, f_dict in dictionary.items():
        norm_data[folderN] = dict()
        norm_data[folderN][signal_shot] = Image_Fitted(f_dict[signal_shot].mimage / 
                                          get_value(f_dict[calibration_shot],attribute,index),do_fit2D)
    return norm_data


# In[ ]:

def normalise_individual_image(dictionary, signal_shot, calibration_shot, attribute, index=None, do_fit2D = False):
    """normalize each image using attribute[index] value - usually 'total' or 'x_data_fit[0]'
        returns constracted dictionary (like what returns 'load_data()' function"""
    norm_data = dict()
    for folderN, f_dict in dictionary.items():
        calibrated_images = []
        for s_elem in f_dict[signal_shot]:
            c_elems = [c_elem for c_elem in f_dict[calibration_shot] if c_elem.shotN == s_elem.shotN]
            if c_elems == []:
                print('s_elem.image_url has no calibration image')
                continue
            calibrated_images = append(calibrated_images, 
                                       Image_Fitted(s_elem.mimage / get_value(c_elems[0],attribute,index), do_fit2D))
        if calibrated_images != []:
            norm_data[folderN] = dict()
            norm_data[folderN][signal_shot] = calibrated_images
    return norm_data

