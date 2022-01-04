import numpy as np
from scipy.interpolate import splrep, splint


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
