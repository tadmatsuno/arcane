import numpy as np
from scipy.interpolate import Akima1DInterpolator, RBFInterpolator
import warnings
from multiprocessing import Pool
from functools import partial

class SpectraGrid:
    def __init__(self, labels, input_values, fluxes, wavelength, no_line_flux = None, ews = None):
        self.labels = labels
        self.input_values = input_values
        self.fluxes = fluxes
        self.wavelength = wavelength
        self.ndim = len(labels)
        self.bounds = [(np.min(input_values[:,i]), np.max(input_values[:,i])) for i in range(self.ndim)]
        self.no_line_flux = no_line_flux
        self.ews = ews
        self.construct_grid()
        
    
    def construct_grid(self):
        if self.ndim == 1:
            sortidx = np.argsort(self.input_values.ravel())
            values_in = self.input_values.ravel()[sortidx]
            fluxes_in = self.fluxes[sortidx]
            self.interpolator = Akima1DInterpolator(values_in, fluxes_in)
            if self.ews is not None:
                ews_in = self.ews[sortidx]
                ews = np.hstack([-ews_in[::-1],0.0,ews_in])
                fluxes = np.vstack([fluxes_in[::-1,:], self.no_line_flux, fluxes_in])
                self.ew2flux = Akima1DInterpolator(ews, fluxes)
                self.ew2input = Akima1DInterpolator(ews_in, values_in)
                self.input2ew = Akima1DInterpolator(values_in, ews_in)
            if self.no_line_flux is not None:
                flux_diff = self.no_line_flux - fluxes_in
                depths0 = np.max(flux_diff, axis=1)
                depths = np.hstack([-depths0[::-1],0.0,depths0])
                minus_flux = flux_diff[::-1,:] + self.no_line_flux
                fluxes = np.vstack([minus_flux, self.no_line_flux, fluxes_in])
                self.depth2flux = Akima1DInterpolator(depths, fluxes)
                self.depth2input = Akima1DInterpolator(depths0, values_in)
                self.input2depth = Akima1DInterpolator(values_in, depths0)
        else:
            warnings.warn("High-dimensional interpolation is experimental.")  
            self.interpolator = RBFInterpolator(np.max(flux_diff, axis=1), self.fluxes)
            if self.ews is not None:
                self.input2ew = RBFInterpolator(self.input_values, self.ews)
    
    def __call__(self, values, depth_interp = False):
        values = np.array(values).reshape(-1,self.ndim)
        if self.ndim == 1:
            values = values.ravel()
            inside_grid = (self.bounds[0][0] <= values) & (values <= self.bounds[0][1])
        else:
            inside_grid = np.all([(b[0] <= values[:,i]) & (values[:,i] <= b[1]) for i,b in enumerate(self.bounds)],axis=0)
        if depth_interp:
            raise ValueError("Not implemented yet")
            if not hasattr(self,'depth_interpolator'):
                results = self.interpolator(values)
                results[:,:] = np.nan    
        else:
            results = self.interpolator(values)
        results[~inside_grid,:] = np.nan
        if (self.ndim == 1) and (len(values)==1):
            results = results[0]
        return self.wavelength,results
            

def _run_fsynth(args):
    fsynth, input_dict = args
    return fsynth(**input_dict)



def construct_grid(fsynth, parameters_name, values, 
        grid_values = None, labels = None, 
        no_line_flux_input = None,
        parallel = True,
        **kwargs):
    """
    Construct a grid of synthetic spectra by varying multiple parameters.

    Parameters
    ----------
    fsynth : function
        A function that generates a synthetic spectrum given parameter values, such as moog.synth, turbospectrum.synth, etc.
        
    parameters_name : list or str
        A list of parameter names to vary. 
        
    values : list or list of lists-like
        A list of values for each parameter. If a single parameter is given, this should be a list of values for that parameter.
        
    grid_values : list with the same shape as values or function that coverts values to grid_values, optional
        A list of values to use for the grid. If None, the same values as `values` are used. 
        This is useful when the values are not directly the parameter values, e.g., linelist with different isotopic fractions.
        
    labels : list or str, optional
        A list of labels for each input.
        
    no_line_flux_input : float, optional
        The input to be used to generate the no-line flux. Only considered when ndim=1
        
    **kwargs : additional keyword arguments to be passed to fsynth
    """
    
    if callable(grid_values):
        grid_values = np.array([grid_values(v) for v in values])
    elif grid_values is None:
        grid_values = values

    shape = np.shape(grid_values)
    if len(shape) == 1:
        ndim = 1
        nlen = shape[0]
        grid_values = np.array(grid_values).reshape((nlen,1))
    elif len(shape) == 2:
        nlen, ndim = shape
    else:
        raise ValueError('grid_values should be 1D or 2D array-like')
    
    assert len(values) == nlen, "Length of values and grid_values should be the same"
    
    if isinstance(parameters_name, str):
        parameters_name = [parameters_name]
    
    if labels is None:
        labels = parameters_name
    if isinstance(labels, str):
        labels = [labels]    
    assert len(labels) == ndim, "Length of labels should be the same as the dimension of grid_values"
    
    if np.ndim(values)==1:
        values = np.array(values).reshape(-1,1)

    if parallel:
        inputs = [dict({name: v for v,name in zip(vv,parameters_name)}, in_parallel = True, **kwargs) for vv in values]
        tasks = [(fsynth, inputd) for inputd in inputs]
        with Pool(2) as pool:
            results_list = pool.map(_run_fsynth, tasks)
    else:
        results_list = [fsynth(**dict({name: v for v,name in zip(vv,parameters_name)}, in_parallel = True, **kwargs)) for vv in values]
        

    for r in results_list:
        assert len(r) == 2, "fsynth should return a tuple of (wavelength, flux)"
        assert len(r[0]) == len(r[1]), "wavelength and flux should have the same length"
        assert np.all(r[0] == results_list[0][0]), "wavelength should be the same for all spectra"
    
    if ndim == 1 and no_line_flux_input is not None:
        input0 = dict({parameters_name[0]: no_line_flux_input, **kwargs})
        result0 = fsynth(**input0)
        assert len(result0) == 2, "fsynth should return a tuple of (wavelength, flux)"
        assert len(result0[0]) == len(result0[1]), "wavelength and flux should have the same length"
        assert np.all(result0[0] == results_list[0][0]), "wavelength should be the same for all spectra"
        no_line_flux = result0[1]
    else:
        no_line_flux = None
    
    return SpectraGrid(
        labels,
        grid_values,
        np.array([res[1] for res in results_list]),
        results_list[0][0],
        no_line_flux=no_line_flux
    )

