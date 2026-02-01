import numpy as np
from arcane.utils import utils, cross_corr
from scipy.interpolate import splev, splrep,interp1d
import warnings
from scipy.optimize import minimize,LinearConstraint,Bounds, curve_fit
import copy
import tqdm
from astropy.constants import c
import multiprocessing as mp
from functools import partial
from dataclasses import dataclass
import functools
ckm = c.to('km/s').value

def _fit4parallel(_model,fparam,xx,yy):
    model = copy.deepcopy(_model)
    model.fit(xx,yy)
    return fparam(model)

def validate(init):
    '''
    This decorator makes sure that each ModelBase subclass implements
    func_model, func_residual, or update methods.
    '''
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._validate()  # defined on Base
    return wrapper

@dataclass
class FittingResults:
    wavelength_raw : np.ndarray
    flux_raw : np.ndarray
    wavelength : np.ndarray
    flux : np.ndarray
    use_flag : np.ndarray
    flux_fit : np.ndarray
    model_parameters : dict
    std_residual : float
    optimization_output : any = None
    result_continuum : any = None
    
class ModelBase:
    def __init__(self, 
        niterate = 10, low_rej = 3., high_rej= 5., grow = 0.05, 
        naverage = 1, fit_mode = 'subtract',
        samples = [], std_from_central = False):
        '''
        Base class for model spectra, which could be a continuum, 
        (an) absorption line(s), synthetic spectra
        
        It needs to have an evaluate method that takes the following arguments:       
        - wavelength : array
        - model_parameters : dict containing model parameters
        One of the following methods also need to be implemented:
        - func_model : function
            A function that takes wavelength and model parameters as input
            and returns the model flux.
            This function is used in the fitting procedure.
        - func_residual : function
            A function that takes wavelength, flux, and model parameters as input
            and returns the residual between model and data.
        - update : function
            A function that takes wavelength and flux as input and update model_parameters.
        Note that not all parameters in self.model_parameters need to be updated. 
        There should be some parameters that control which parameters are updated.
        
                
        The fit function of this class performs fitting together with sigma-clipping.
        Once fitting is conducted, wavelength and flux of the data are stored as attributes.
        '''
        self.niterate = niterate
        self.low_rej = low_rej
        self.high_rej = high_rej
        self.grow = grow
        self.naverage = naverage
        self.samples = samples
        self.fit_mode = fit_mode
        self.std_from_central = std_from_central

    def __init_subclass__(self, **kwargs):
        super().__init_subclass__(**kwargs)

        # Only wrap if subclass defines its own __init__
        init = self.__dict__.get("__init__")
        if init is not None:
            self.__init__ = validate(init)

    def _validate(self):
        '''
        Validate that the subclass has implemented required methods.
        '''
        if not hasattr(self, "evaluate"):
            raise NotImplementedError("evaluate method must be implemented in the subclass.")
        elif not callable(self.evaluate):
            raise NotImplementedError("evaluate method must be callable.")
        
        if hasattr(self, "func_model"):
            if not callable(self.func_model):
                raise ValueError("func_model should be a callable function.")
            if not hasattr(self, "x0_default"):
                raise ValueError("x0_default attribute must be defined when func_model is implemented.")
            if not hasattr(self, "bounds_default"):
                raise ValueError("bounds_default attribute must be defined when func_model is implemented.")
            if not hasattr(self, "scales"):
                raise ValueError("scales attribute must be defined when func_model is implemented.")
        elif hasattr(self, "func_residual"):
            if not callable(self.func_residual):
                raise ValueError("func_residual should be a callable function.")
            if not hasattr(self, "x0_default"):
                raise ValueError("x0_default attribute must be defined when func_residual is implemented.")
            if not hasattr(self, "bounds_default"):
                raise ValueError("bounds_default attribute must be defined when func_residual is implemented.")
        elif hasattr(self, "update"):
            if not callable(self.update):
                raise ValueError("update should be a callable function.")
        else:
            raise NotImplementedError("Either func_model, func_residual, or update method must be implemented in the subclass.")
                
    def __call__(self, wavelength, continuum_model = None):
        if continuum_model is not None:
            return self.evaluate(wavelength,self.model_parameters) * continuum_model(wavelength)
        else:
            return self.evaluate(wavelength,self.model_parameters)

    def uncertainty(self, fparameter, std, nmc=100, iid = True, parallel = False):
        '''
        This function determines the uncertainty of the model by 
        conducting MC sampling.
        If iid is False, the effect of the continuum placement is considered.
        i.e., every pixel is shifted by the same amount.
        If iid is True, the effect of photon noise in each pixel is considered.
        
        The user must provide photon noise level or continuum placement error as std.

        NOTE: this function adds error on self.flux. If you do not want to double-
        count errors, make sure self.flux is a noise-free data (such as the best-fit model). 

        Parameters
        ----------
        fparameter : function
            A function that returns the parameters of interest.
        
        nmc: int
            if nmc = 0 and iid = False, a MC sampling won't be performed.
            Instead, the error is estimatead by displacing the continuum +- 1, 2, and 3 sigma.
            The result will be a tuple, -3, -2, -1, 1, 2, 3
        '''
        parallel = False # For now, parallel is not supported
        
        model0 = copy.deepcopy(self)
        mc_out = []
        flux0 = self.flux.copy()
        if iid:
            fluxes = [flux0+std*np.random.randn(len(self.flux)) for _ in range(nmc)]
        else:
            if nmc > 0:
                fluxes = [flux0+std*np.random.randn() for _ in range(nmc)]
            else:
                fluxes = [flux0+std*_ for _ in [-3,-2,-1,1,2,3]]
        if parallel:
            with mp.Pool() as pool:
                mc_out = list(tqdm.tqdm(pool.map(partial(_fit4parallel,self,fparameter,self.wavelength),fluxes)))
        else:
            for flux in tqdm.tqdm(fluxes):
                self.fit(self.wavelength,flux)
                mc_out.append(fparameter(self))
        #if iid:
        #    for ii in tqdm.tqdm(range(nmc)):
        #        self.fit(self.wavelength,flux0+std*np.random.randn(len(self.flux)))
        #        mc_out.append(fparameter(self))
        #else:
        #    if nmc > 0:
        #        for ii in tqdm.tqdm(range(nmc)):
        #            self.fit(self.wavelength,flux0+std*np.random.randn())
        #            mc_out.append(fparameter(self))
        #    else:
        #        for nsigma in [-3,-2,-1,1,2,3]:
        #            self.fit(self.wavelength,flux0+std*nsigma)
        #            mc_out.append(fparameter(self))
        mc_result = {key: [d[key] for d in mc_out] for key in mc_out[0]}
        self = copy.deepcopy(model0)
        return mc_result

    def fit(self, wavelength, flux, continuum_model = None, optimization_parameter = {}, debug=False):
        '''
        Fit the model to the data with sigma-clipping.
        Parameters
        ----------
        wavelength : array
            Wavelength of the data
        flux : array
            Flux of the data
        Returns
        -------
        FittingResults dataclass
        '''
        result_continuum = None
        
        wvl, flx = utils.average_nbins(self.naverage, wavelength, flux)
        use_flag = utils.get_region_mask(wvl, self.samples) & (np.isfinite(flx))
        if debug:
            print(f"Initial number of points used for fitting: {np.sum(use_flag)} out of {len(wvl)}")
        outliers = np.array([False]*len(wvl))
        for _ in range(self.niterate):
            if debug:
                print(f'{_} th iteration: Fitting with {np.sum(use_flag)} points out of {len(wvl)}'+\
                    f" the number of outliers in the last iteration: {np.sum(outliers)}")
                print(f"Among use_flag, postive flux: {np.sum(flx[use_flag] > 0)}, negative flux: {np.sum(flx[use_flag] <= 0)}"+\
                    f", nan flux: {np.sum(flx[use_flag] == 0)}")
            use_flag = use_flag & (~outliers)
            # Try curve_fit -> minimize -> self.update
            if hasattr(self, "func_model"):
                p0 = self.x0_default
                bounds = self.bounds_default
                x_scale = self.scales if self.scales is not None else np.ones(len(p0))
                if continuum_model is not None:
                    if debug:
                        print("Fitting with continuum model using curve_fit")
                    def func_model(wvl, *xi):
                        youter = self.func_model(wavelength,*xi)
                        if debug:
                            print(f"Current parameters: {xi}")
                        result_cont = continuum_model.fit(wavelength, flux / youter ,debug=debug)
                        ycontinuum = continuum_model(wvl)
                        return self.func_model(wvl, *xi) * ycontinuum
                else:
                    if debug:
                        print("Fitting without continuum model using curve_fit")                     
                    func_model = self.func_model
                opt_out = curve_fit(
                    func_model,
                    wvl[use_flag],
                    flx[use_flag],
                    p0 = p0,
                    bounds = bounds,
                    x_scale = x_scale,
                    method = "trf",
                    nan_policy = "omit",
                    **optimization_parameter
                )
                youter = self.func_model(wavelength, *opt_out[0])
                if continuum_model is not None:
                    result_continuum = continuum_model.fit(wavelength, flux / youter, debug=debug)
                _ = self.func_model(wvl[use_flag], *opt_out[0])
            elif hasattr(self, "func_residual"):
                p0 = self.x0_default
                bounds = self.bounds_default
                flx_in = flx[use_flag]
                if continuum_model is not None:
                    if debug:
                        print("Fitting with continuum model using minimize")
                    youter = self(wvl)
                    result_continuum = continuum_model.fit(wavelength, flux / youter, debug=debug)
                    flx_in /= continuum_model(wvl[use_flag])
                elif debug:
                    print("Fitting without continuum model using minimize")
                opt_out = minimize(
                    lambda x: np.sum(
                    self.func_residual(wvl[use_flag], flx_in, x)**2),
                    self.x0_default,
                    method = "Nelder-Mead",
                    bounds = Bounds(*bounds),
                    **optimization_parameter
                )
                _ = self.func_residual(wvl[use_flag], flx_in, opt_out.x)
            else:
                flx_in = flx[use_flag]
                if continuum_model is not None:
                    if debug:
                        print("Fitting with continuum model using update (deterministic)")
                    youter = self(wvl)
                    result_continuum = continuum_model.fit(wavelength, flux / youter, debug=debug)
                    flx_in /= continuum_model(wvl[use_flag])
                elif debug:
                    print("Fitting without continuum model using update (deterministic)")
                model_parameters, opt_out = self.update(
                    wvl[use_flag], flx_in)
            
            yfit = self(wvl, continuum_model = continuum_model)
            model_parameters = self.model_parameters.copy()
            if self.fit_mode == 'subtract':
                outliers = utils.sigmaclip(
                    wvl, flx,
                    use_flag, yfit,
                    self.grow, self.low_rej, self.high_rej,
                    std_from_central = self.std_from_central)                 
            elif self.fit_mode == 'ratio':
                outliers = utils.sigmaclip(
                    wvl, flx/yfit,
                    use_flag, np.ones(len(wvl)),
                    self.grow, self.low_rej,self.high_rej,
                    std_from_central = self.std_from_central)
            else:
                raise ValueError('fit_mode has to be either of ratio or subtract')
            if np.sum(outliers & use_flag) / np.sum(use_flag) > 0.95:
                warnings.warn('More than 95% of points {0:d}/{1:d} are rejected. No further sigma-clipping is applied.'.format(\
                    np.sum(outliers & use_flag), np.sum(use_flag)))
                outliers = np.array([False]*len(wvl))
                break
            if np.sum(use_flag & outliers) == 0:
                # No more points to remove
                break
        
        yfit = yfit
        residual_std = np.nanstd(flx[use_flag]-yfit[use_flag])
        
        return FittingResults(
            wavelength_raw = wavelength,
            flux_raw = flux,
            wavelength = wvl,
            flux = flx,
            use_flag = use_flag,
            flux_fit = yfit,
            model_parameters = model_parameters,
            std_residual = residual_std,
            optimization_output = opt_out,
            result_continuum = result_continuum
            )
            
        
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
    
    def evaluate(self, xx, model_parameters):
        if model_parameters['spl'] is None:
            raise ValueError('Call update or set scipy spline class object')
        else:
            return splev(xx, model_parameters['spl'])

    def update(self, xx, yy):
        model_parameters = {"spl": self._spline3fit(xx, yy, self.dx_knots)}
        self.model_parameters = model_parameters
        return model_parameters, None

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
            niterate = niterate, low_rej = low_rej, high_rej = high_rej, grow = grow,
            naverage = naverage, fit_mode = fit_mode, samples = samples, 
            std_from_central = std_from_central)
        self.dx_knots = dwvl_knots
        self.model_parameters = {'spl':None}


class ContinuumPolynomial(ModelBase):

    def update(self, xx, yy):
        model_parameters = {"poly": np.polynomial.Polynomial.fit(xx, yy, deg=self.order)}
        self.model_parameters = model_parameters
        return model_parameters, None

    def evaluate(self, xx, model_parameters):
        if model_parameters['poly'] is None:
            raise ValueError('Call update or set np.polynomial.Polynomial class object')
        else:
            return model_parameters['poly'](xx)

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
            
        model_parameters = self.model_parameters.copy()
        def f_residual(xi):
            for ii,map1 in enumerate(xi_map):
                label, sindices = map1.split('_')
                assert label in ['dwvl','depth','fwhm','fgfwhm'], 'xi_map specifies unknown parameters'
                for sidx in sindices.split(','):
                    idx = int(sidx)
                    model_parameters[label][idx] = xi[ii]
            if self.fit_control['constrain_gaussian']:
                for ii,map1 in enumerate(xi_map):
                    label, sindices = map1.split('_')
                    if label != 'fwhm':
                        continue
                    for sidx in sindices.split(','):
                        idx = int(sidx)
                        model_parameters['fwhm'][idx] = xi[ii]*model_parameters['fgfwhm'][ii]
            return np.sum((self.evaluate(xx, model_parameters) - yy)**2.)*(self.snr)**2
        res = minimize(f_residual, x0=x0, \
            bounds= Bounds(const_low,const_high),options={'ftol':0.01/len(xx)})
        residual = f_residual(res.x)
        self.model_parameters = model_parameters
        return model_parameters, res


    def evaluate(self, xx, model_parameters):
        return utils.voigts_multi_fwhm_fgfwhm(\
            model_parameters['center'] + model_parameters['dwvl'],
            model_parameters['depth'],
            model_parameters['fwhm'],
            model_parameters['fgfwhm'])(xx)
        
    # I don't think it is a bad idea to return the updated model_parameters in the following methods
    def delete_line(self,idx):
        self.nlines -= 1
        self.model_parameters['center'] = np.delete(self.model_parameters['center'],idx)
        self.model_parameters['dwvl'] = np.delete(self.model_parameters['dwvl'],idx)
        self.model_parameters['fwhm'] = np.delete(self.model_parameters['fwhm'],idx)
        self.model_parameters['fgfwhm'] = np.delete(self.model_parameters['fgfwhm'],idx)
        return self.model_parameters

    def _add_line_base(self):
        self.nlines +=1 
        self.model_parameters['center'] = np.append(self.model_parameters['center'],0.0)
        self.model_parameters['dwvl'] = np.append(self.model_parameters['dwvl'],0.0)
        self.model_parameters['fwhm'] = np.append(self.model_parameters['fwhm'],self.initial_fwhm)
        self.model_parameters['fgfwhm'] = np.append(self.model_parameters['fgfwhm'],1.0)
        return self.model_parameters

    def add_line(self, central_wavelength, depth, sigma, gamma):
        fwhm,fgfwhm,flfwhm = utils.gslg2wxgxl(sigma,gamma)
        self.add_line2(central_wavelength, depth, fwhm, fgfwhm)
        return self.model_parameters

    def add_line2(self, central_wavelength, depth = -1, fwhm = -1, fgfwhm = -1):
        self._add_line_base()
        self.model_parameters['center'][-1] = central_wavelength
        if depth >= 0.0:
            self.model_parameters['depth'][-1] = depth
        if fwhm >= 0.0:
            self.model_parameters['fwhm'][-1] = fwhm
        if fgfwhm >= 0.0:
            self.model_parameters['fgfwhm'][-1] = fgfwhm        
        return self.model_parameters

    def add_line_from_ew(self, central_wavelength, ew, sigma, gamma):
        fwhm,fgfwhm,flfwhm = utils.gslg2wxgxl(sigma,gamma)
        self.add_line_from_ew2(central_wavelength, ew, fwhm, fgfwhm)
        return self.model_parameters

    def add_line_from_ew2(self, central_wavelength, ew, fwhm = -1, fgfwhm = -1):
        self._add_line_base()
        self.model_parameters['center'][-1] = central_wavelength
        if fwhm >= 0.0:
            self.model_parameters['fwhm'][-1] = fwhm
        if fgfwhm >= 0.0:
            self.model_parameters['fgfwhm'][-1] = fgfwhm        
        self.model_parameters['depth'][-1] = \
            utils.voigt_depth_fwhm_fgfwhm(ew,self.model_parameters['fwhm'][-1],self.model_parameters['fgfwhm'][-1])
        return self.model_parameters
    # The comment on returning model_parameters applies to the methods down here

    def get_ews(self):
        return [utils.voigt_EW_fwhm_fgfwhm(\
            self.model_parameters['depth'][ii], 
            self.model_parameters['fwhm'][ii], 
            self.model_parameters['fgfwhm'][ii]) for ii in range(self.nlines)]

    def __init__(self, central_wavelengths, 
        initial_depth = 0.3, initial_fwhm = 0.1,
        niterate = 10, low_rej = 3., high_rej = 5., grow = 0.05,
        naverage = 1, fit_mode = 'subtract', snr = 50.,
        samples = [], std_from_central = False, 
        **kwargs):
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
            if key in kwargs.keys():
                value_in = kwargs[key]
            if key in ['fix_dwvl','fix_fwhm','voigt']:
                if len(np.atleast_1d(value_in)) == 1:
                    self.fit_control[key] = np.atleast_1d(value_in).repeat(self.nlines)
                else:
                    assert len(value_in) == self.nlines, f'Length mismatch between central_wavelengths and {key}'
                    self.fit_control[key] = np.atleast_1d(value_in)
            else:
                self.fit_control[key] = value_in
        for key in kwargs.keys():
            if not key in self.default_fit_control.keys():
                warnings.warn(f'{key} is ignored since it is not a valid keyword for fit_control')


class LineSynth(ModelBase):
    fitting_parameters_default = {\
        "scaling" : {},
        "bounds" : {},
        "diff_step" : 1e-3,
        "curve_fit_kwargs" : {}
    }
    def evaluate(self,xx, model_parameters):
        wvl, flux = self.fsynth(**{\
            key: model_parameters[key] for key in self.synth_parameter_names}
        )
        smoothed = utils.smooth_spectrum(wvl, flux, model_parameters['vFWHM'])
        flx_out =utils.rebin(wvl * (1.0 + model_parameters['rv'] / ckm), # Doppler shift
            smoothed, 
            xx,
            conserve_count=False)    
        return flx_out

    def func_model(self, wvl, *xi):
        model_parameters = self.model_parameters.copy()
        for x,key in zip(xi,self.parameters_to_fit):
            if key.startswith('I_'):# Isotope
                model_parameters[key] = 10.**x
            else:# rv, vFWHM, or abundances
                model_parameters[key] = x
        self.model_parameters = model_parameters
        return self.evaluate(wvl, model_parameters)
        

    def __init__(self, 
        fsynth, synth_parameters, parameters_to_fit, vfwhm_in = 5.0,
        rv_in = 0.0,
        niterate = 10, low_rej = 3., high_rej = 5., grow = 0.05,
        naverage = 1, fit_mode = 'subtract',
        samples = [], std_from_central = False,
        continuum_model = None
        ):
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
        
        rv_in : real
            Initial guess of the radial velocity in km/s

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
            niterate = niterate, low_rej = low_rej, high_rej = high_rej, grow = grow,
            naverage = naverage, fit_mode = fit_mode,
            samples = samples, std_from_central = std_from_central)
        self.fsynth = fsynth
        self.synth_parameter_names = list(synth_parameters.keys())
        self.model_parameters = {'vFWHM':vfwhm_in, "rv":rv_in,**synth_parameters}
        for key in parameters_to_fit:
            assert key in self.model_parameters.keys(), f'{key} must be vFWHM, "rv" or included in synth_parameters'
        self.parameters_to_fit = parameters_to_fit
        self.fitting_result = None
        
        if continuum_model is not None:
            self.continuum_model = continuum_model

        # Default initial guess, bounds, and scaling
        x0 = []
        bounds = []
        
        for key in self.parameters_to_fit:
            if key.startswith('I_'): # Isotope
                x0.append(np.log10(self.model_parameters[key]))
            else: # rv, vFWHM, or abundances
                x0.append(self.model_parameters[key])
    
            if key == "vFWHM":
                bounds.append( (0.0,np.inf) )
            else:
                bounds.append( (-np.inf,np.inf) )                
        
        self.x0_default = x0
        self.bounds_default = np.array(bounds).T
        self.scales = None    
        
