from scipy.signal import correlate,correlation_lags
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from arcane_dev.utils import utils
from astropy.constants import c
ckm  = c.to('km/s').value

def measure_vshift(wvl,flx1,flx2,max_shift=1000,wvl2=None)\
    -> float:
    '''
    Measure the velocity shift between two spectra.
    The flx2 is estimated to be redshifted by vshift relative to flx1.
    (so divide wvl2 by (1+vshift/ckm) to correct for redshift)

    Parameters
    ----------
    wvl : array
        Wavelength array of the first spectrum.
    flx1 : array
        Continuum normalized flux array of the first spectrum.
    flx2 : array
        Continuum normalized flux array of the second spectrum.
    max_shift : int, optional
        Maximum velocity shift to consider in km/s. 
        The default is 1000 km/s.
    wvl2 : array, optional 
        Wavelength array of the second spectrum. 
        If not provided, the wavelength array of the first spectrum is used.
    
        
    Returns
    -------
    vshift : float
        Velocity shift in km/s.
    '''
    # convert wvl so that it is evenly spaced in log scale
    wvl0 = np.logspace(np.log10(wvl[0]+0.01),np.log10(wvl[-1]-0.01),len(wvl))
    dlogwvl = np.median(np.diff(np.log10(wvl0)))
    delv = ckm*(10.**dlogwvl-1.0)
    flx1 = utils.rebin(wvl,flx1,wvl0,conserve_count=False)

    if wvl2 is not None:
        flx2 = utils.rebin(wvl2,flx2,wvl0,conserve_count=False)
    else:
        flx2 = utils.rebin(wvl,flx2,wvl0,conserve_count=False)
    flx1 = 1.0-flx1
    flx2 = 1.0-flx2

    ycor = correlate(flx1,flx2,mode='same')
    xcor = correlation_lags(len(flx1),len(flx2),mode='same')*delv
    mask = np.abs(xcor)<max_shift
    xcor = xcor[mask]
    ycor = ycor[mask]
    fcor = interp1d(xcor,ycor,bounds_error=False,fill_value=0.0)
    res = minimize(lambda x: -fcor(x),xcor[np.argmax(ycor)])
    vshift = res.x[0]
#    return xcor,ycor,vshift
    return -vshift
