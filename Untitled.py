
# coding: utf-8

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


# In[ ]:

class Image_Basics():
    """Basic image processing, with only center determination and background substructing. 
        isgood parameter is needed for indicating if smth wrong with image during it's processing 
        (i.e. if image is blank)"""
    def __init__(self,image, image_url='not_defined'):
        self.image_url = image_url
        self.image = image
        c_pos = self.image.argmax()
        self.center_pos = (c_pos//self.image.shape[1], c_pos%self.image.shape[1])
        self.image = self.bgd_substract()
        self.total = sum(self.image)
        self.isgood = True
    def bgd_substract(self, slice_to_c = (-20,-1)):
        """ Substracts background, which is calculated using vertical strip at right side"""
        data_for_bgd_det = self.image[:,slice_to_c[0]:slice_to_c[1]]
        av = sum(data_for_bgd_det,1)/data_for_bgd_det.shape[1]
        return self.image - tile(av[:,newaxis],self.image.shape[1])
    def fit_gaussian1D(self, axis, width=10):
        from scipy.optimize import curve_fit
        data = sum(self.image, axis)
        popt, pcov = curve_fit(gaussian, range(len(data)), data, p0=(self.total, argmax(data), width, 0))
        return popt
    def fit_gaussian2D(self):
        """Returns (height, y, x, width_y, width_x)
        the gaussian parameters of a 2D distribution found by a fit"""
        from scipy import optimize
        params = (self.total, self.fit1D_y[1], self.fit1D_x[1], self.fit1D_y[2], self.fit1D_x[2], 0)
        errorfunction = lambda p: ravel(gaussian2D(*p)(*indices(self.image.shape)) -
                                 self.image)
        popt, success = optimize.leastsq(errorfunction, params)
        return popt

