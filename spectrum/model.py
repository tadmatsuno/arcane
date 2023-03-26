import numpy as np
from arcane_dev.utils import utils
from scipy.interpolate import splev, splrep
import warnings
from scipy.optimize import minimize,LinearConstraint,Bounds


class ModelBase:
    def __init__(self, fbase, fupdate,
        niterate = 10, low_rej = 3., high_rej= 5., grow = 0.05, 
        naverage = 1, fit_mode = 'subtract',
        samples = [], std_from_central = False):
        '''
        Base class for model spectra, which could be a continuum, 
        (an) absorption line(s), synthetic spectra
        '''
        self.fbase = fbase
        self.fupdate = fupdate
        self.niterate = niterate
        self.low_rej = low_rej
        self.high_rej = high_rej
        self.grow = grow
        self.naverage = naverage
        self.samples = samples
        self.fit_mode = fit_mode
        self.std_from_central = std_from_central
    
    def __call__(self):
        if hasattr(self,'wavelength'):
            return self.fbase(self.wavelength)
        else:
            print('Fit data first!')


    def fit(self, wavelength, flux):
        self.wavelength, self.flux = \
            utils.average_nbins(self.naverage, wavelength, flux)
        self.use_flag = utils.get_region_mask(self.wavelength, self.samples)
        outliers = np.array([False]*len(self.wavelength))
        for _ in range(self.niterate):
            self.use_flag = self.use_flag & (~outliers)
            self.fupdate(
                self.wavelength[self.use_flag], self.flux[self.use_flag])
            yfit = self.fbase(self.wavelength)
            if self.fit_mode == 'subtract':
                outliers = utils.sigmaclip(
                    self.wavelength, self.flux,\
                    self.use_flag, yfit,\
                    self.grow, self.low_rej, self.high_rej,
                    std_from_central = self.std_from_central)                 
            elif self.fit_mode == 'ratio':
                outliers = utils.sigmaclip(
                    self.wavelength, self.flux/yfit,\
                    self.use_flag, np.ones(len(self.wavelength)),\
                    self.grow, self.low_rej,self.high_rej,
                    std_from_central = self.std_from_central)
            else:
                raise ValueError('fit_mode has to be either of ratio or subtract')
        self.yfit = yfit
  
class ContinuumSpline3(ModelBase):
    def _get_npt_btw_knots(self, xx, knots):
        '''
        Counts number of points between knots

        Parameters
        ----------
        xx : array

        knots : array
        '''
        nknots = len(knots)
        npoint = len(xx)
        npt_btw_knots = np.sum(\
        ((xx - knots[:-1].repeat(len(xx)).reshape(nknots-1,npoint)) * 
        (knots[1:].repeat(len(xx)).reshape(nknots-1,npoint)-xx))>=0,\
        axis=1)
        return npt_btw_knots

    def _spline3fit(self, xx, yy, dx_knots):
        '''
        Spline fitting to a spectrum defined by xx and yy.
        The interval between knots needs to be pre-defined. However,
        if there are too few points between the knots, some knots can be 
        removed.

        Parameters
        ----------
        xx : array

        yy : array

        dx_knots : float
            the interval between knots.

        Returns 
        -------
        spl : scipy.interpolate.splrep 
        '''
        knots = np.linspace(xx[0],xx[-1],\
            np.maximum(int((xx[-1]-xx[0])//dx_knots),3))
        knots_in = knots[1:-1]
        ## Remove knots with no points aound them
        npt_btw_knots = self._get_npt_btw_knots(xx,knots)
        knots_in = knots_in[(npt_btw_knots[1:]>=1)|(npt_btw_knots[:-1]>=1)]

        ## Combine knots between which there are no points
        npt_btw_knots = self._get_npt_btw_knots(xx,\
            np.hstack([knots[0],knots_in,knots[-1]]))
        if any(npt_btw_knots<1):
            knots2 = [knots[0]]
            ii = 1
            while (ii+1<len(npt_btw_knots)):
                if (npt_btw_knots[ii] >= 1):
                    knots2.append(knots_in[ii-1])
                    ii = ii +1
                else:
                    x1,x2 = knots_in[ii-1],knots_in[ii]
                    n1,n2 = npt_btw_knots[ii-1],npt_btw_knots[ii+1]
                    knots2.append((x1*n1+x2*n2)/(n1+n2))
                    ii = ii + 2
            try:
                knots2.append(knots_in[ii-1])
            except:
                pass
            knots2.append(knots[-1])
            knots2 = np.array(knots2)
        else:
            knots2 = np.hstack([knots[0],knots_in,knots[-1]])
        if len(knots2)<=2:
            knots2 = np.array([0.5*(knots[0]+knots[-1])])
        return splrep(xx,yy,task=-1,t=knots2[1:-1])
    
    def evaluate(self, xx):
        if self.model_parameters['spl'] is None:
            raise ValueError('Call update or set scipy spline class object')
        else:
            return splev(xx, self.model_parameters['spl'])

    def update(self, xx, yy):
        self.model_parameters['spl'] = self._spline3fit(xx, yy, self.dx_knots)
        return

    def __init__(self, \
        dwvl_knots, 
        niterate = 10, low_rej = 3., high_rej = 5., grow = 0.05,
        naverage = 1, fit_mode = 'ratio',
        samples = [], std_from_central = False):
        '''
        Continuum expressed with cubic spline
        Meant to be equivalent to iraf's continuum task

        Parameters
        ----------

        dwvl_knots : 
            Spacing between knots
            If there are no data points between knots, there locations are adjusted.

        niterate : int 
            The number of iterations for sigma clipping

        low_rej : real
            The lower threshold for rejection. 
            Points with normalized flux < 1.0 - std*low_rej are removed

        high_rej  : real
            The higher threshold for rejection. 
            Points with normalized flux < 1.0 + std*high_rej are removed

        grow : real
            Data points within [grow] wavelength within rejected points 
            are also removed

        naverage : int
            Binning

        fit_mode : str
            either 'subtract' or 'ratio'
            If ratio is selected, sigma clipping is done on y / yfit - 1.0

        samples : list of list
            Regions used for fitting.
            example: [[ 4000, 4010], [4020, 4030]] 
                Data points between 4000 and 4010 A, and between 4020--4030 A are used.
            
        std_from_central : 
            When estimating sigma for rejection, one can only use the central part of 
            spectrum. This is useful when a spectrum becomes noisy at the edges.

        '''
        super().__init__(
            self.evaluate, self.update,
            niterate = niterate, low_rej = low_rej, high_rej = high_rej, grow = grow,
            naverage = naverage, fit_mode = fit_mode, samples = samples, 
            std_from_central = std_from_central)
        self.dx_knots = dwvl_knots
        self.model_parameters = {'spl':None}


class ContinuumPolynomial(ModelBase):

    def update(self, xx, yy):
        self.model_parameters['poly'] = \
            np.polynomial.Polynomial.fit(xx, yy, deg=self.order)

    def evaluate(self, xx):
        if self.model_parameters['poly'] is None:
            raise ValueError('Call update or set np.polynomial.Polynomial class object')
        else:
            return self.model_parameters['poly'](xx)

    def __init__(self, order, 
        niterate = 10, low_rej = 3., high_rej = 5., grow = 0.05,
        naverage = 1, fit_mode = 'subtract',
        samples = [], std_from_central = False):
        '''
        Continuum expressed with a polynomical function

        Parameters
        ----------

        Order : int
            The order of polynomial

        niterate : int 
            The number of iterations for sigma clipping

        low_rej : real
            The lower threshold for rejection. 
            Points with normalized flux < 1.0 - std*low_rej are removed

        high_rej  : real
            The higher threshold for rejection. 
            Points with normalized flux < 1.0 + std*high_rej are removed

        grow : real
            Data points within [grow] wavelength within rejected points 
            are also removed

        naverage : int
            Binning

        fit_mode : str
            either 'subtract' or 'ratio'
            If ratio is selected, sigma clipping is done on y / yfit - 1.0


        samples : list of list
            Regions used for fitting.
            example: [[ 4000, 4010], [4020, 4030]] 
                Data points between 4000 and 4010 A, and between 4020--4030 A are used.
            
        std_from_central : 
            When estimating sigma for rejection, one can only use the central part of 
            spectrum. This is useful when a spectrum becomes noisy at the edges.

        '''
        super().__init__(
            self.evaluate, self.update,
            niterate = niterate, low_rej = low_rej, high_rej = high_rej, grow = grow,
            naverage = naverage, fit_mode = fit_mode,
            samples = samples, std_from_central = std_from_central)
        self.order = order
        self.model_parameters = {'poly':None}

class LineProfile(ModelBase):
    default_fit_control = {\
        'share_dwvl': False, 'fix_dwvl': False, 'max_dwvl' : 0.1,
        'constrain_gaussian' : False,  
        'share_fwhm': False, 'fix_fwhm': False, 'max_fwhm' : 10.0, 'min_fwhm' : 0.01,
        'voigt' : True}

    def update(self, xx, yy):
        if any(self.fit_control['voigt']):
            voigt_tmp = self.fit_control['voigt'].copy()
            self.fit_control['voigt'] = np.array([False]*self.nlines)
            self.update(xx,yy)
            self.fit_control['voigt'] = voigt_tmp.copy()

        xi_map = [] # How to map the input of f_residual (xi) to model_parameters
        x0 = [] # initial guess 
        # The following matrix and two arrays will be used in LinearConstraint
        const_low = []
        const_high = []      
        if self.fit_control['share_dwvl']: # Use the same dwvl value for lines with fix_dwvl = False 
            if not all(self.fit_control['fix_dwvl']):# if dwvl is fixed for all the lines, nothing will be done
                xi_map_tmp = 'dwvl_'
                for ii in range(self.nlines):
                    if not self.fit_control['fix_dwvl'][ii]:
                        xi_map_tmp += f'{ii},'
                xi_map.append(xi_map_tmp[:-1])
                x0.append(self.model_parameters['dwvl'][0])
                const_low.append( -self.fit_control['max_dwvl'])
                const_high.append(self.fit_control['max_dwvl'])
        else:
            for ii in range(self.nlines):
                if not self.fit_control['fix_dwvl'][ii]:
                    xi_map.append(f'dwvl_{ii}')
                    x0.append(self.model_parameters['dwvl'][ii])
                    const_low.append( -self.fit_control['max_dwvl'])
                    const_high.append(self.fit_control['max_dwvl'])
        # FWHM
        if self.fit_control['share_fwhm']:
            if not all(self.fit_control['fix_fwhm']): # If fwhm is fixed for all the lines nothing will be done
                for ii in range(self.nlines):
                    xi_map_tmp = 'fwhm_'
                    for ii in range(self.nlines):
                        if not self.fit_control['fix_fwhm'][ii]:
                            xi_map_tmp += f'{ii}'
                    xi_map.append(xi_map_tmp[:-1])
                    if self.fit_control['constrain_gaussian']:
                        x0.append(self.model_parameters['fwhm'][0]*\
                            self.model_parameters['fgfwhm'][0])
                    else:
                        x0.append(self.model_parameters['fwhm'][0])
                    const_low.append(self.fit_control['min_fwhm'])
                    const_high.append(self.fit_control['max_fwhm'])
        else:
            for ii in range(self.nlines):
                if not self.fit_control['fix_fwhm'][ii]:
                    xi_map.append(f'fwhm_{ii}')
                    if self.fit_control['constrain_gaussian']:
                        x0.append(self.model_parameters['fwhm'][ii]*\
                            self.model_parameters['fgfwhm'][ii])
                    else:
                        x0.append(self.model_parameters['fwhm'][ii])
                    const_low.append(self.fit_control['min_fwhm'])
                    const_high.append(self.fit_control['max_fwhm'])
        # Voigt
        for ii in range(self.nlines):
            if self.fit_control['voigt'][ii]:
                xi_map.append(f'fgfwhm_{ii}')
                x0.append(self.model_parameters['fgfwhm'][ii])
                const_low.append(0.0)
                const_high.append(1.0)
        # depth
        for ii in range(self.nlines):
            xi_map.append(f'depth_{ii}')
            x0.append(self.model_parameters['depth'][ii])
            const_low.append(-np.inf)
            const_high.append(np.inf)

        def f_residual(xi):
            for ii,map1 in enumerate(xi_map):
                label, sindices = map1.split('_')
                assert label in ['dwvl','depth','fwhm','fgfwhm'], 'xi_map specifies unknown parameters'
                for sidx in sindices.split(','):
                    idx = int(sidx)
                    self.model_parameters[label][idx] = xi[ii]
            if self.fit_control['constrain_gaussian']:
                for ii,map1 in enumerate(xi_map):
                    label, sindices = map1.split('_')
                    if label != 'fwhm':
                        continue
                    for sidx in sindices.split(','):
                        idx = int(sidx)
                        self.model_parameters['fwhm'][idx] = xi[ii]*self.model_parameters['fgfwhm'][ii]
            return np.sum((self.evaluate(xx) - yy)**2.)*(self.snr)**2
        res = minimize(f_residual, x0=x0, \
            bounds= Bounds(const_low,const_high),options={'ftol':0.01/len(xx)})
        residual = f_residual(res.x)


    def evaluate(self, xx):
        return utils.voigts_multi_fwhm_fgfwhm(\
            self.model_parameters['center'] + self.model_parameters['dwvl'],
            self.model_parameters['depth'],
            self.model_parameters['fwhm'],
            self.model_parameters['fgfwhm'])(xx)

    def delete_line(self,idx):
        self.nlines -= 1
        self.model_parameters['center'] = np.delete(self.model_parameters['center'],idx)
        self.model_parameters['dwvl'] = np.delete(self.model_parameters['dwvl'],idx)
        self.model_parameters['fwhm'] = np.delete(self.model_parameters['fwhm'],idx)
        self.model_parameters['fgfwhm'] = np.delete(self.model_parameters['fgfwhm'],idx)

    def _add_line_base(self):
        self.nlines +=1 
        self.model_parameters['center'] = np.append(self.model_parameters['center'],0.0)
        self.model_parameters['dwvl'] = np.append(self.model_parameters['dwvl'],0.0)
        self.model_parameters['fwhm'] = np.append(self.model_parameters['fwhm'],self.initial_fwhm)
        self.model_parameters['fgfwhm'] = np.append(self.model_parameters['fgfwhm'],1.0)

    def add_line(self, central_wavelength, depth, sigma, gamma):
        fwhm,fgfwhm,flfwhm = utils.gslg2wxgxl(sigma,gamma)
        self.add_line2(central_wavelength, depth, fwhm, fgfwhm)

    def add_line2(self, central_wavelength, depth = -1, fwhm = -1, fgfwhm = -1):
        self._add_line_base()
        self.model_parameters['center'][-1] = central_wavelength
        if depth >= 0.0:
            self.model_parameters['depth'][-1] = depth
        if fwhm >= 0.0:
            self.model_parameters['fwhm'][-1] = fwhm
        if fgfwhm >= 0.0:
            self.model_parameters['fgfwhm'][-1] = fgfwhm        

    def add_line_from_ew(self, central_wavelength, ew, sigma, gamma):
        fwhm,fgfwhm,flfwhm = utils.gslg2wxgxl(sigma,gamma)
        self.add_line_from_ew2(central_wavelength, ew, fwhm, fgfwhm)

    def add_line_from_ew2(self, central_wavelength, ew, fwhm = -1, fgfwhm = -1):
        self._add_line_base()
        self.model_parameters['center'][-1] = central_wavelength
        if fwhm >= 0.0:
            self.model_parameters['fwhm'][-1] = fwhm
        if fgfwhm >= 0.0:
            self.model_parameters['fgfwhm'][-1] = fgfwhm        
        self.model_parameters['depth'][-1] = \
            utils.voigt_depth_fwhm_fgfwhm(ew,self.model_parameters['fwhm'][-1],self.model_parameters['fgfwhm'][-1])

    def get_ews(self):
        return [utils.voigt_EW_fwhm_fgfwhm(\
            self.model_parameters['depth'][ii], 
            self.model_parameters['fwhm'][ii], 
            self.model_parameters['fgfwhm'][ii]) for ii in range(self.nlines)]

    def __init__(self, central_wavelengths, 
        initial_depth = 0.3, initial_fwhm = 0.1,
        niterate = 10, low_rej = 3., high_rej = 5., grow = 0.05,
        naverage = 1, fit_mode = 'subtract', snr = 50.,
        samples = [], std_from_central = False, kw_fit_control = {}):
        '''
        Continuum expressed with a polynomical function

        Parameters
        ----------

        nline : int
            The number of lines

        niterate : int 
            The number of iterations for sigma clipping

        low_rej : real
            The lower threshold for rejection. 
            Points with normalized flux < 1.0 - std*low_rej are removed

        high_rej  : real
            The higher threshold for rejection. 
            Points with normalized flux < 1.0 + std*high_rej are removed

        grow : real
            Data points within [grow] wavelength within rejected points 
            are also removed

        naverage : int
            Binning

        fit_mode : str
            either 'subtract' or 'ratio'
            If ratio is selected, sigma clipping is done on y / yfit - 1.0

        samples : list of list
            Regions used for fitting.
            example: [[ 4000, 4010], [4020, 4030]] 
                Data points between 4000 and 4010 A, and between 4020--4030 A are used.
            
        std_from_central : 
            When estimating sigma for rejection, one can only use the central part of 
            spectrum. This is useful when a spectrum becomes noisy at the edges.

        '''
        super().__init__(
            self.evaluate, self.update,
            niterate = niterate, low_rej = low_rej, high_rej = high_rej, grow = grow,
            naverage = naverage, fit_mode = fit_mode,
            samples = samples, std_from_central = std_from_central)
        self.snr = snr
        self.initial_depth = initial_depth
        self.initial_fwhm = initial_fwhm
        self.nlines = len(np.atleast_1d(central_wavelengths))
        self.model_parameters = \
            {'center' : np.atleast_1d(central_wavelengths),
            'dwvl' : np.zeros(self.nlines),
            'depth' : np.zeros(self.nlines) + initial_depth,
            'fwhm' : np.zeros(self.nlines) + initial_fwhm,
            'fgfwhm' : np.zeros(self.nlines)+1.0}
        self.fit_control = {}
        for key in self.default_fit_control.keys():
            value_in = self.default_fit_control[key]
            if key in kw_fit_control.keys():
                value_in = kw_fit_control[key]
            if key in ['fix_dwvl','fix_fwhm','voigt']:
                if len(np.atleast_1d(value_in)) == 1:
                    self.fit_control[key] = np.atleast_1d(value_in).repeat(self.nlines)
                else:
                    assert len(value_in) == self.nlines, f'Length mismatch between central_wavelengths and {key}'
                    self.fit_control[key] = np.atleast_1d(value_in)
            else:
                self.fit_control[key] = value_in
        for key in kw_fit_control.keys():
            if not key in self.default_fit_control.keys():
                warnings.warn(f'{key} is ignored since it is not a valid keyword for kw_fit_control')



class LineSynth(ModelBase):
    def evaluate(self,xx):
        wvl, flux = self.fsynth(**self.synth_parameters)
        smoothed = utils.smooth_spectrum(wvl,flux,self.model_parameters['vFWHM'])
        return utils.rebin(wvl,smoothed,xx,conserve_count=False)

    def update(self,xx,yy):
        x0 = []
        bounds = []
        for key in self.update_synth_parameters:
            if key.startswith('I_'):
                x0.append(np.log10(self.synth_parameters[key]))
            else:
                x0.append(self.synth_parameters[key])
            bounds.append((None,None))
        if not self.fit_control['fix_vFWHM']:
            x0.append(self.model_parameters['vFWHM'])
            bounds.append((0.0,None))
        initial_simplex = np.array([np.array(x0)]*(len(x0)+1))
        for ii in range(len(x0)):
            initial_simplex[ii+1,ii] += 0.1
        def f_residual(xi):
            for x,key in zip(xi,self.update_synth_parameters):
                if key.startswith('I_'):
                    self.synth_parameters[key] = 10.**x
                else:
                    self.synth_parameters[key] = x
            if not self.fit_control['fix_vFWHM']:
                # FWHM is also one of the fitting parameters
                self.model_parameters['vFWHM'] = xi[-1]
            return np.sum((self.evaluate(xx)-yy)**2.)
        res = minimize(f_residual,
            x0 = x0, bounds = bounds,
            options={'initial_simplex':initial_simplex,'xatol':0.1},\
            method='Nelder-Mead')
        residual = f_residual(res.x)

    def __init__(self, 
        fsynth, synth_parameters, parameters_to_fit, vfwhm_in = 5.0,
        niterate = 10, low_rej = 3., high_rej = 5., grow = 0.05,
        naverage = 1, fit_mode = 'subtract',
        samples = [], std_from_central = False, kw_fit_control = {'fix_vFWHM':True}):
        '''
        This class is to fit a synthetic spectrum to an observed spectrum
        If only one parameter, excluding vFWHM, is to be fit, use LineSynth1param,
        which is much faster thanks to the use of interpolation.
        
        Parameters
        ----------
        fsynth : function
            A function that returns a synthetic spectrum. See synthesis.moog and synthesis.turbospectrum
        
        synth_parameters : dict
            Parameters for fsynth. Usually, model atmosphere, linelist, and abundances are needed.
        
        parameters_to_fit : list of str
            Parameters to be fit. These parameters must be included in the keys of synth_parameters.
        
        vfwhm_in : real
            Initial guess of the instrumental FWHM in km/s

        niterate : int
            Number of iterations for sigma clipping
        
        low_rej : real
            The lower threshold for rejection.
        
        high_rej : real
            The upper threshold for rejection.

        grow : real
            Data points within [grow] wavelength within rejected points will also be rejected.

        naverage : int
            Binning

        fit_mode : str
            either 'subtract' or 'ratio'

        samples : list of list
            Regions used for fitting.

        kw_fit_control : dict
            Keywords for fitting control, for example whether to fix vFWHM or not.
        '''

        super().__init__(
            self.evaluate, self.update,
            niterate = niterate, low_rej = low_rej, high_rej = high_rej, grow = grow,
            naverage = naverage, fit_mode = fit_mode,
            samples = samples, std_from_central = std_from_central)
        for key in parameters_to_fit:
            assert key in synth_parameters.keys(), f'{key} must be included in synth_parameters'
        self.fsynth = fsynth
        self.synth_parameters = synth_parameters
        self.update_synth_parameters = parameters_to_fit
        self.model_parameters = {'vFWHM':vfwhm_in}
        self.fit_control = kw_fit_control.copy()

class LineSynth1param(ModelBase):
    def evaluate(self,xx , force_recompute = False):
        if force_recompute or (self.grid_size == 0.0):
            wvl, flux = self.fsynth(**self.synth_parameters)
        else:
            x1 = self.synth_parameters[self.update_synth_parameter] // self.grid_size
            x2 = x1 + 1
            if not x1 in self.grid.keys():
                dict_synth_tmp = self.synth_parameters.copy()
                dict_synth_tmp[self.update_synth_parameter] = self.grid_size * x1
                wvl,flux = self.fsynth(**dict_synth_tmp)
                self.grid[x1] = flux
                self.grid['wvl'] = wvl
            if not x2 in self.grid.keys():
                dict_synth_tmp = self.synth_parameters.copy()
                dict_synth_tmp[self.update_synth_parameter] = self.grid_size * x2
                wvl,flux = self.fsynth(**dict_synth_tmp)
                self.grid[x2] = flux
                self.grid['wvl'] = wvl
            wvl = self.grid['wvl'] 
            f1 = (x2 * self.grid_size - self.synth_parameters[self.update_synth_parameter])/self.grid_size
            f2 = 1. - f1
            flux = f1 * self.grid[x1] + f2 * self.grid[x2]
        smoothed = utils.smooth_spectrum(wvl,flux,self.model_parameters['vFWHM'])
        return utils.rebin(wvl,smoothed,xx,conserve_count=False)

    def update(self,xx,yy):
        if self.update_synth_parameter.startswith('I_'):
            # If isotpe ratio, use log scale
            x0 = [np.log10(self.synth_parameters[self.update_synth_parameter])]
            xatol = 0.1
        else:
            x0 = [self.synth_parameters[self.update_synth_parameter]]
            xatol = 0.001
        bounds = [(None,None)]
        if not self.fit_control['fix_vFWHM']:
            x0.append(self.model_parameters['vFWHM'])
            bounds.append((0.0,None))
        initial_simplex = np.array([np.array(x0)]*(len(x0)+1))
        for ii in range(len(x0)):
            initial_simplex[ii+1,ii] += 0.1
        def f_residual(xi):
            if self.update_synth_parameter.startswith('I_'):
                self.synth_parameters[self.update_synth_parameter] = 10.**xi[0]
            else:
                self.synth_parameters[self.update_synth_parameter] = xi[0]
            if not self.fit_control['fix_vFWHM']:
                # FWHM is also one of the fitting parameters
                self.model_parameters['vFWHM'] = xi[1]
            residual = np.sum((self.evaluate(xx,self.fit_control['force_recompute'])-yy)**2.)*(self.snr**2.0)
            return residual
        res = minimize(f_residual,
            x0 = x0, bounds = bounds,
            options={'initial_simplex':initial_simplex,'xatol':xatol},\
            method='Nelder-Mead')
        residual = f_residual(res.x)

    def __init__(self, 
        fsynth, synth_parameters, parameter_to_fit, vfwhm_in = 5.0,
        grid_size = 0.1, grid_scale = 'linear', snr=50.,
        niterate = 10, low_rej = 3., high_rej = 5., grow = 0.05,
        naverage = 1, fit_mode = 'subtract',
        samples = [], std_from_central = False, 
        kw_fit_control = {'fix_vFWHM':True}):
        '''
        This class is to fit a synthetic spectrum to an observed spectrum.
        Only one parameter can be fit. If more than one parameter is to be fit, use LineSynth.
        Note that linear interpolation is adopted.

        Parameters
        ----------
        fsynth : function
            A function that returns a synthetic spectrum. See synthesis.moog and synthesis.turbospectrum
        
        synth_parameters : dict
            Parameters for fsynth. Usually, model atmosphere, linelist, and abundances are needed.
        
        parameter_to_fit : str
            The parameter to be fit. The parameter must be included in the keys of synth_parameters.
        
        vfwhm_in : real
            Initial guess of the instrumental FWHM in km/s
        
        grid_size : real
            The grid size for interpolation
        
        grid_scale : str
            The scale of the grid. Either 'linear' or 'log'

        snr : real
            The signal-to-noise ratio of the synthetic spectrum. 
            
        niterate : int
            Number of iterations for sigma clipping
        
        low_rej : real
            The lower threshold for rejection.
        
        high_rej : real
            The upper threshold for rejection.

        grow : real
            Data points within [grow] wavelength within rejected points will also be rejected.

        naverage : int
            Binning

        fit_mode : str
            either 'subtract' or 'ratio'

        samples : list of list
            Regions used for fitting.

        kw_fit_control : dict
            Keywords for fitting control, for example whether to fix vFWHM or not.
            fix_vFWHM : bool
                Whether to fix vFWHM or not.
        '''
        fit_control_default = {'fix_vFWHM':True, 'force_recompute':False}
        
        super().__init__(
            self.evaluate, self.update,
            niterate = niterate, low_rej = low_rej, high_rej = high_rej, grow = grow,
            naverage = naverage, fit_mode = fit_mode,
            samples = samples, std_from_central = std_from_central)
        assert parameter_to_fit in synth_parameters.keys(), f'{parameter_to_fit} must be included in synth_parameters'
        self.fsynth = fsynth
        self.snr = snr
        self.synth_parameters = synth_parameters
        self.update_synth_parameter = parameter_to_fit
        self.model_parameters = {'vFWHM':vfwhm_in}
        self.grid_size = grid_size
        assert grid_scale in ['linear','log'], 'grid_scale must be either linear or log'
        self.grid_scale = grid_scale
        self.fit_control = fit_control_default.copy()
        for key,val in kw_fit_control.items():
            self.fit_control[key] = val
        if self.grid_size == 0.0:
            self.fit_control['force_recompute'] = True
        self.grid = {}