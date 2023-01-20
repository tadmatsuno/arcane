import numpy as np
from scipy.interpolate import splrep, splint
from scipy.special import voigt_profile
from astropy.constants import c
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter
ckm = c.cgs.value*1.0e-5

c1 = 0.5346
c2 = (1.0-c1)**2.
g1 = 2.0*np.sqrt(2.0*np.log(2.0))


def get_dx(x):
  '''
  This function computes the bin size from an array of x-values.
  dx_i = (dx_{i+1} - dx_{i-1}) / 2,
  dx_0 = dx_1 and dx_n = dx_{n-1}.

  Parameters
  ----------
  x : list of float

  Returns
  -------
  dx : list of float
  '''
  dx1 = x[1:]-x[:-1]
  dx = np.hstack([dx1[0],0.5*(dx1[1:]+dx1[:-1]),dx1[-1]])
  return dx

def average_nbins(nbin,x,y):
  '''
  This function carris out binning. 
  The integral of y over x will be conserved. 

  Parameters
  ----------
  nbin : int

  x : list of float

  y : list of float

  Returns
  -------
  xx : list of float

  yy : list of float

  '''
  if nbin==1:
    return x,y
  else:
    nfin = len(x)//nbin*nbin
    dx = get_dx(x)
    xx = np.sum([(x*dx)[0+ii:nfin:nbin] for ii in range(nbin)],axis=0)/\
      np.sum([dx[0+ii:nfin:nbin] for ii in range(nbin)],axis=0)
    yy = np.sum([(y*dx)[0+ii:nfin:nbin] for ii in range(nbin)],axis=0)/\
      np.sum([dx[0+ii:nfin:nbin] for ii in range(nbin)],axis=0)
    return xx,yy

def rebin(x,y,xnew,conserve_count=True): 
  '''
  This function conducts re-binning of a spectrum.

  Paramaters
  ----------
  x : list of float
    x-values of the input spectrum

  y : list of float
    y-values (flux) of the input spectrum
  
  xnew : list of float
    x-values of the output spectrum
    flux at xnew will be computed

  conserve_count : bool
    True in default. 
    If True, total count is conserved so the sum(ynew) will be 
    the same as sum(y). Useful when flux is shown in photon counts.
    If False, intergration is conserved. 

  Returns
  -------
    ynew : list of float
      y-values at xnew
  '''

  dx = get_dx(x) 
  dxnew = get_dx(xnew) 
  if conserve_count: # total count conserved (input is per pix)
      spl = splrep(x,y/dx,k=1,task=0,s=0) 
      return np.array([splint(xn-0.5*dxn,xn+0.5*dxn,spl) \
        for xn,dxn in zip(xnew,dxnew)]) 
  else: #total flux conserved (input is in physical unit)
        #use this for normalized spectra
      spl = splrep(x,y,k=1,task=0,s=0) 
      return np.array([splint(xn-0.5*dxn,xn+0.5*dxn,spl)/dxn \
        for xn,dxn in zip(xnew,dxnew)]) 

def x_sorted(xx,yy):
  '''
  This function sort yy in ascending order in xx

  Parameters:
  xx : 1d array
  yy : 1d or 2d-array

  '''

  argxx = np.argsort(xx)
  try:
    len(yy[0])
    return (xx[argxx],)+tuple([y[argxx] for y in yy])
  except:
    return xx[argxx],yy[argxx]

def get_region_mask(x,samples):
  '''
  this function returns the mask reflecting the sampling region defined
  by the samples parameter

  Parameters
  ----------
  x : array

  samples : list of list

  '''
  if len(samples)==0:
    return np.isfinite(x)
  else:
    return np.any([(x-ss[0])*(x-ss[1])<=0 for ss in samples],axis=0)

def sigmaclip(xx,yy,use_flag,yfit,grow,low_rej,high_rej,
  std_from_central = False):
  '''
  This function does a sigma clipping.
  Note that the edge is not allowed to be removed.
  When computing the standard deviation, only the central 50% part of 
  the spectrum will be considered. 
  Returns an array of bool, which is True if the point is sigma-clipped.

  Parameters
  ----------
  xx : array

  yy : array

  use_flag : mask 
    only points where use_mask is True are considered

  yfit : array
    results of fitting to yy

  grow : float 
    in wavelength unit.
    Points within "grow" from removed points in sigma-clipping will 
    also be removed
  
  low_rej, high_rej : float, float
    Threshold for sigma-clipping. 

  std_from_central : bool
    if True, only central 50% points will be used to estimate residual scatter

  Returns
  -------
  removemask : array


  '''
  resid = yy-yfit
  if std_from_central:
    quat = np.sum(use_flag)//4 
    ystd = np.nanstd(resid[use_flag][quat:-quat],ddof=1)
  else:
    ystd = np.nanstd(resid[use_flag],ddof=1)

  outside = (resid < (-ystd*low_rej)) | (resid > (ystd*high_rej))
#  xoutside = xx[outside]
#  xoutside = xoutside.repeat(len(xx)).reshape(len(xoutside),len(xx))
#  removemask = np.any(((xoutside-grow)<xx)&(xx<(xoutside+grow)),axis=0)
  kd_outside = KDTree(np.atleast_2d(xx[outside]).T)
  r_outside, idx_outside = kd_outside.query(np.atleast_2d(xx).T, workers=-1)
  removemask = r_outside <= grow
  removemask[np.min(np.nonzero(use_flag))] = False
  removemask[np.max(np.nonzero(use_flag))] = False
  return removemask

def wxg2wgwl(fwhm,fgfwhm):
  '''
  get fwhms of Gaussian and Lorenzian components from FWHM and gFWHM/FWHM

  Parameters
  ----------
  fwhm : float
  
  fgfwhm :float
    FWHM_G/FWHM

  '''
  flfwhm = (c1-np.sqrt(c1**2. - (2.*c1-1.)*(1.0-fgfwhm**2.)) )/ (2.*c1-1.)
  lfwhm = fwhm*np.maximum(np.minimum(flfwhm,1.0),0.0)
  gfwhm = fwhm*fgfwhm
  return gfwhm,lfwhm 

def wxl2wgwl(fwhm,flfwhm):
  '''
  get fwhms of Gaussian and Lorenzian components from FWHM and flFWHM

  Parameters
  ----------
  fwhm : float
  
  flfwhm :float
    FWHM_L/FWHM = 2*gamma/FWHM

  '''
  lfwhm = fwhm*flfwhm
  fgfwhm = np.sqrt( (2.*c1-1.)*flfwhm**2.0 - 2.*c1*flfwhm + 1)
  gfwhm = fwhm*np.maximum(np.minimum(fgfwhm,1.0),0.0)
  return gfwhm,lfwhm

def gs2wg(sigma):
  '''
  Get fwhm of a gaussian profile
  '''
  return g1*sigma

def wg2gs(fwhm):
  '''
  Get sigma of a gaussian profile from fwhm
  '''
  return fwhm/g1

def lg2wl(gamma):
  '''
  Get fwhm of a Lorenzian profile
  '''
  return 2.*gamma

def wl2lg(fwhm):
  '''
  Get gamma of a Lorenzian profile from fwhm
  '''
  return fwhm/2.

def gslg2wxgxl(sigma,gamma):
  '''
  get fwhm of a Voigt profile

  Returns
  ----------
  fwhm : float

  fgfwhm :float
    FWHM_G/FWHM

  flfwhm :float
    FWHM_L/FWHM = 2*gamma/FWHM

  '''
  lfwhm = lg2wl(gamma)
  gfwhm = gs2wg(sigma)
  return wgwl2wxgxl(gfwhm,lfwhm)

def wgwl2wxgxl(gfwhm,lfwhm):
  '''
  get fwhm of a Voigt profile

   Returns
  ----------
  fwhm : float

  fgfwhm :float
    FWHM_G/FWHM

  flfwhm :float
    FWHM_L/FWHM = 2*gamma/FWHM

  '''
  fwhm = c1*lfwhm + np.sqrt(c2*lfwhm**2. + gfwhm**2.)
  flfwhm = lfwhm/fwhm
  fgfwhm = gfwhm/fwhm
  return fwhm,fgfwhm,flfwhm

def wxl2gslg(fwhm,flfwhm):
  '''
  get sigma of Gaussian and gamma of Lorenzian components from FWHM and flFWHM

  Parameters
  ----------
  fwhm : float
  
  flfwhm :float
    FWHM_L/FWHM = 2*gamma/FWHM

  '''
  gfwhm,lfwhm = wxl2wgwl(fwhm,flfwhm)
  sigma = wg2gs(gfwhm)
  gamma = wl2lg(lfwhm)
  return sigma,gamma

def wxg2gslg(fwhm,fgfwhm):
  '''
  get sigma of Gaussian and gamma of Lorenzian components from FWHM and fgFWHM

  Parameters
  ----------
  fwhm : float
  
  fgfwhm :float
    FWHM_G/FWHM

  '''
  gfwhm,lfwhm = wxg2wgwl(fwhm,fgfwhm)
  sigma = wg2gs(gfwhm)
  gamma = wl2lg(lfwhm)
  return sigma,gamma

def voigt_EW_sigma_gamma(depth,sigma,gamma):
  '''
  Returns the EW of a Voigt profile
  '''
  return depth/voigt_profile(0,sigma,gamma)

def voigt_EW_fwhm_flfwhm(depth,fwhm,flfwhm):
  '''
  Returns the EW of a Voigt profile
  '''
  sigma,gamma = wxl2gslg(fwhm,flfwhm)
  return voigt_EW_sigma_gamma(depth,sigma,gamma)

def voigt_EW_fwhm_fgfwhm(depth,fwhm,fgfwhm):
  '''
  Returns the EW of a Voigt profile
  '''
  sigma,gamma = wxg2gslg(fwhm,fgfwhm)
  return voigt_EW_sigma_gamma(depth,sigma,gamma)

def voigt_depth_sigma_gamma(ew,sigma,gamma):
  '''
  Returns the depth of a Voigt profile
  '''
  return ew*voigt_profile(0,sigma,gamma)

def voigt_depth_fwhm_flfwhm(ew,fwhm,flfwhm):
  '''
  Returns the depth of a Voigt profile
  '''
  sigma,gamma = wxl2gslg(fwhm,flfwhm)
  return voigt_depth_sigma_gamma(ew,sigma,gamma)

def voigt_depth_fwhm_fgfwhm(ew,fwhm,fgfwhm):
  '''
  Returns the depth of a Voigt profile
  '''
  sigma,gamma = wxg2gslg(fwhm,fgfwhm)
  return voigt_depth_sigma_gamma(ew,sigma,gamma)


def voigts_multi_sigma_gamma(x0s,depths,sigmas,gammas):
  '''
  Returns sum of multiple Voigt profiles.

  Parameters
  ----------
  x0s : float or 1d-array
    the central wavelength of the lines 

  depths : float or 1d-array
    the depths of the lines 

  sigmas : float or 1d-array
    sigma in Gaussian components 

  gammas : float or 1d-array
    gamma in Lorenzian components

  Returns
  -------
  function that returns flux at given position
    
  '''
  x0s = np.atleast_1d(x0s)
  depths = np.atleast_1d(depths)
  sigmas = np.abs(np.atleast_1d(sigmas))
  gammas = np.abs(np.atleast_1d(gammas))
  return lambda x:\
    np.sum([dep*voigt_profile(x-x0,ss,ll)/voigt_profile(0,ss,ll) \
      for x0,dep,ss,ll in zip(x0s,depths,sigmas,gammas)],axis=0)

def voigts_multi_fwhm_flfwhm(x0s,depths,fwhms,flfwhm):
  '''
  Returns sum of multiple Voigt profiles.

  Parameters
  ----------
  x0s : float or 1d-array
    the central wavelength of the lines 

  depths : float or 1d-array
    the depths of the lines 

  fwhms : float or 1d-array
    FWHM of the lines 

  flfwhm : float or 1d-array
    FWHM_L/FWHM = 2*gamma/FWHM

  Returns
  -------
  function that returns flux at given position
    
  '''    
  sigma,gamma = wxl2gslg(fwhms,flfwhm)
  return voigts_multi_sigma_gamma(x0s,depths,sigma,gamma)

def voigts_multi_fwhm_fgfwhm(x0s,depths,fwhms,fgfwhm):
  '''
  Returns sum of multiple Voigt profiles.

  Parameters
  ----------
  x0s : float or 1d-array
    the central wavelength of the lines 

  depths : float or 1d-array
    the depths of the lines 

  fwhms : float or 1d-array
    FWHM of the lines 

  fgfwhm : float or 1d-array
    FWHM_G/FWHM 

  Returns
  -------
  function that returns flux at given position
    
  '''    
  print(x0s,depths,fwhms,fgfwhm)
  sigma,gamma = wxg2gslg(fwhms,fgfwhm)
  return voigts_multi_sigma_gamma(x0s,depths,sigma,gamma)

def textsamples(samples,reverse=False):
  ''' 
  Convert 2 x N sample region definition to text 
  or vice versa. 

  Parameters
  ----------
  samples : 2 x N list, or str
    An example for the list is [[5000,5100], [5150, 5250]]
    An example for str is 
      5000 5100
      5150 5250
  reverse : bool
    If True, the conversion is from text to list
  '''

  if reverse:
    if samples.count('\n') == 0:
      samples = [samples]
    else:
      samples = samples.split('\n')
    outpair = []
    for ss in samples:
      if len(ss.lstrip().rstrip()) == 0:
        continue
      ss = ss.lstrip().rstrip()
      outpair.append([float(ss.split()[0]),float(ss.split()[1])])
    return outpair
  else:
    outtxt = ''
    for ss in samples:
      outtxt += '{0:10.3f} {1:10.3f}\n'.format(ss[0],ss[1])
    return outtxt

def get_grid_value(grid_points,target, outside='nearest'):
  if np.min(grid_points)>target:
    if outside == 'nearest':
      return np.min(grid_points),np.min(grid_points),False
    else:
      raise ValueError(\
        f'The input value is too low. available:{np.min(grid_points)} target={target}')
  if np.max(grid_points)<target:
    if outside == 'nearest':
      return np.max(grid_points),np.max(grid_points),False
    else:
      raise ValueError(\
        f'The input value is too high. available:{np.max(grid_points)} target={target}')
  if (np.min(grid_points)==target)|(np.max(grid_points)==target):
    return target,target,True
  p1 = np.max(grid_points[grid_points<target])
  p2 = np.min(grid_points[grid_points>=target])
  if p2==target:
    return p2,p2,True
  else:
    return p1,p2,True

def smooth_spectrum(wvl,flx,vfwhm):
  wvlmin,wvlmax = min(wvl),max(wvl)
  wc = (wvlmin+wvlmax)/2.0
  dvel = (wvlmax-wvlmin)/(len(wvl)-1.0)/wc*ckm
  vsigma = vfwhm/ (np.sqrt(8.0*np.log(2)))
  psigma = vsigma/dvel
  flout = gaussian_filter(flx,psigma,mode='constant',cval=1.0,truncate=10.0)
  return flout
