import numpy as np
import os
from numpy.core.fromnumeric import std
from scipy.interpolate import splev,splrep

import matplotlib.pyplot as plt
for key in plt.rcParams.keys(): # To avoid conflicts with default shortcuts, which are all disabled.
  if key.startswith('keymap'):
    [plt.rcParams[key].remove(ss) for ss in plt.rcParams[key]]
import matplotlib
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.Qt import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import MouseButton
from .pyqtcontinuum import Ui_Dialog
matplotlib.use('Qt5Agg')
from ..utils import utils


class ContinuumFit:
  '''
  Class to store continuum function
  '''


  def __init__(self,func='spline3',dwvl_knots=6.,niterate=10,
    low_rej=3.,high_rej=5.,grow=0.05,naverage=1,samples=[]):
    '''
    Once you run ContinuumFit.continuum, you have access to 
    the blaze function through ContinuumFit.flx_continuum and
    the normalized flux through ContinuumFit.flx_normalized.

    Parameters
    ----------
    func : str
      The function used to fit the continuum
      Only spline3 (cubic spline) is supported now. 
    
    dwvl_knots : float
      The iterval between knots in wavelength for cubic spline.

    niterate : int
      The number of itereations for sigma-clipping

    low_rej, high_rej : float, float
      Threshold for sigma-clipping. 

    grow : float 
      in wavelength unit.
      Points within "grow" from removed points in sigma-clipping will 
      also be removed

    naverage : int
      When S/N is poor, one can conduct further binning by specifying a 
      number greater than 1.

    samples : list of list
      list storing information about the region used for continuum fitting.
      An example is [[4000,4010],[4015,4025]]. In this case, a spectral 
      range in 4000-4010 and 4015-4025 will be used for fitting.
    '''

    self.func = func
    self.dwvl_knots = dwvl_knots
    self.niterate = int(niterate)
    self.low_rej = low_rej
    self.high_rej = high_rej
    self.grow = grow
    self.naverage = int(naverage)
    self.samples = samples

  def continuum(self,wavelength,flux):
    '''
    Fit continuum of the spectrum defined by wavelength and flux.
    
    Parameters
    ----------
    wavelength : array

    flux : array

    '''

    self.wavelength,self.flux = \
      utils.average_nbins(self.naverage,wavelength,flux)
    self.use_flag = utils.get_region_mask(self.wavelength,self.samples)
    outliers = np.array([False]*len(self.wavelength))
    for ii in range(self.niterate):
      self.use_flag = self.use_flag & (~outliers)
      if self.func=='spline3':
        spl = self._spline3fit(\
          self.wavelength[self.use_flag],
          self.flux[self.use_flag],
          self.dwvl_knots)
        self.spline3tck = spl
        y_cont = splev(self.wavelength,self.spline3tck)
      else:
        raise AttributeError('{0:.s} is not implemented'.format(self.func))
      outliers = utils.sigmaclip(\
        self.wavelength,self.flux/y_cont,\
        self.use_flag,np.ones(len(self.wavelength)),\
        self.grow,\
        self.low_rej,self.high_rej,
        std_from_central = True) ## Fitting might not be good at the edge
    self.knotsx = spl[0]
    self.knotsy = splev(self.knotsx,spl)
    self.flx_continuum = y_cont
    self.flx_normalized = self.flux / self.flx_continuum


  def _spline3fit(self,xx,yy,dx_knots):
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

    knots = np.linspace(xx[0],xx[-1],int((xx[-1]-xx[0])//dx_knots))
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
    spl = splrep(xx,yy,task=-1,t=knots2[1:-1])
    return spl

  def _get_npt_btw_knots(self,xx,knots):
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

class PlotCanvas(FigureCanvas):
  '''
  Parent class for main window
  
  '''

  def __init__(self,parent,layout):
    self.fig,self.axes = plt.subplots(1,1)
    FigureCanvas.__init__(self,self.fig)

    layout.addWidget(self)

    self.line_obs, = self.axes.plot([0],[0],'C7-',lw=0.5) # All observed points
    self.pt_use, = self.axes.plot([0],[0],'C0o',ms=2.0) # Only used points
    self.line_cont, = self.axes.plot([0],[0],'C1-',lw=1.0) # Fitting result
    self.pt_knots, = self.axes.plot([0],[0],'C1o',ms=3.0) # Location of the knots
    self.cursorx = self.axes.axvline(x=0,linestyle='-',color='C7',lw=0.5) # cursor position x
    self.cursory = self.axes.axhline(y=0,linestyle='-',color='C7',lw=0.5) # cursor position y

    self.fig.canvas.mpl_connect('motion_notify_event',self.mouse_move)
    self.txt = self.axes.text(0.0,0.0,'',transform=self.fig.transFigure,
      horizontalalignment='left',verticalalignment='bottom') # display mouse position
    self.mode_txt = self.axes.text(1.0,1.0,'Normal',\
      transform=self.fig.transFigure,
      fontsize='x-large',color='k',
      horizontalalignment='right',verticalalignment='top')  # display the current mode

    self.setFocusPolicy(Qt.ClickFocus)
    self.setFocus()
    toolbar = NavigationToolbar(self ,parent)
    layout.addWidget(self.toolbar)

    self.fig.tight_layout()
    self.updateGeometry()
  

  def mouse_move(self,event):
    '''
    Displays cursor position
    '''
    x,y = event.xdata,event.ydata
    self.cursorx.set_xdata(x)
    self.cursory.set_ydata(y)
    if not ((x is None)|(y is None)):
      self.txt.set_text('x={0:10.3f}    y={1:10.5f}'.format(x,y))
    self.draw()

class MainWindow(QWidget,Ui_Dialog):
  '''
  Class for main window
  One can also specify the parameters for continuum fitting
  '''

  def __init__(self,parent=None,**CFit_kwargs):
    '''
    Parameters
    ----------
    ** CFit_kwargs : parameters for continuum fitting
    
    '''
    super(MainWindow,self).__init__(parent)

    self.CFit = ContinuumFit(**CFit_kwargs)
    self.ui = Ui_Dialog()
    self.ui.setupUi(self)
    
    self.ui.button_fit.installEventFilter(self) # Button for fitting
    self.ui.button_draw.installEventFilter(self) # Button for redrawing
    self.ui.edit_function.installEventFilter(self) # Text box for type of function in continuum fitting
    self.ui.edit_knots.installEventFilter(self) # Text box for knot interval
    self.ui.edit_niter.installEventFilter(self) # Text box for number of iteration
    self.ui.edit_grow.installEventFilter(self) # Text box for grow radius
    self.ui.edit_lowrej.installEventFilter(self) # Text box for low_rej threshod in sigma-clipping
    self.ui.edit_highrej.installEventFilter(self) # Text box for high_rej threshod in sigma-clipping
    self.ui.edit_nave.installEventFilter(self) # Text box for rebinning
    self.ui.edit_samples.installEventFilter(self) # Text box for sampling range

    self.ui.edit_function.setText(self.CFit.func)
    self.ui.edit_knots.setText(\
      '{0:.3f}'.format(self.CFit.dwvl_knots))
    self.ui.edit_lowrej.setText(\
      '{0:.2f}'.format(self.CFit.low_rej))
    self.ui.edit_highrej.setText(\
      '{0:.2f}'.format(self.CFit.high_rej))
    self.ui.edit_niter.setText(\
      '{0:d}'.format(self.CFit.niterate))
    self.ui.edit_nave.setText(\
      '{0:d}'.format(self.CFit.naverage))  
    self.ui.edit_grow.setText(\
      '{0:.3f}'.format(self.CFit.grow))
    self.ui.edit_samples.setPlainText(\
      utils.textsamples((self.CFit.samples)))

    self.ui.main_figure.layout
    self.canvas  =  PlotCanvas(self.ui.left_grid,self.ui.main_figure)

    self.mpl_status = None
    self.canvas.mpl_connect('key_press_event',self.on_press)
#    self.canvas.mpl_connect('pick_event',self.on_pick) # For future implementation of rejection/addition of a data point

  # Input functions
  def input_data(self,wavelength,flux,output=None):
    '''
    The basic function to set input data. 

    Parameters
    ----------
    wavelength : 1d-array

    flux : 1d-array

    output : str
      filename for the output normalized spectrum
    '''
    self.wavelength = wavelength
    self.flux = flux
    self.CFit.continuum(self.wavelength,self.flux)
    def getminmax(xx):
      xmax = np.max(xx)
      xmin = np.min(xx)
      dx = xmax-xmin
      return xmin-dx*0.05,xmax+dx*0.05
    self.canvas.axes.set_xlim(getminmax(self.wavelength))
    self.canvas.axes.set_ylim(getminmax(self.flux))
    self.draw_fig()
    if not output is None:
      self.output = output


  def input_multi_data(self,multi_wavelength,multi_flux,output_multi_head=None,output=None):
    '''
    Set input data for multispec spectra

    Parameters
    ----------
    multi_wavelength : 1d-or 2d-array

    multi_flux : 1d-or 2d-array

    output_multi_head : str
      the header for file names of output of individual orders
    
    output : str
      filename for the output 1d normalized spectrum
  
    '''
    try:
      len(multi_wavelength[0])
      pass
    except:
      multi_wavelength = [multi_wavelength]
      multi_flux = [multi_flux]
    assert len(multi_wavelength)==len(multi_flux),'Wavelengths and fluxs have different numbers of orders'
    nptx = np.array([len(ws) for ws in multi_wavelength])
    npty = np.array([len(fs) for fs in multi_flux])
    assert all(nptx==npty),'Wavelength and flux have different numbers of points'
    
    wvlidx = np.argsort([np.min(ws) for ws in multi_wavelength])### Sort in wavelength
    wvl_tmp,multi_wavelength = multi_wavelength,[]
    flx_tmp,multi_flux = multi_flux,[]
    for idx in wvlidx:
      multi_wavelength.append(wvl_tmp[idx])
      multi_flux.append(flx_tmp[idx])

    self.multi_wavelength = multi_wavelength
    self.multi_flux = multi_flux
    self.norder = len(multi_wavelength)
    self.current_order = 0
    self.multi_blaze = []
    self.multi_normalized = []
    self.input_data(\
      self.multi_wavelength[self.current_order],
      self.multi_flux[self.current_order],output=None)
    if not output is None:
      self.output = output 
    if not output_multi_head is None:
      self.output_multi_head  = output_multi_head
  
  def input_long1d(self,long1d_wavelength,long1d_flux,\
      wvl_block=100.,wvl_overlap=20.,output=None):
    '''
    Set input data for long-1d spectrum. 
    Spectrum is split into small chunks, each of which has a range of wvl_block. 
    Adjecent chuncks have overlapping region of wvl_overlap.

    Parameters
    ----------
    long1d_wavelength : 1d-array

    long1d_flux : 1d-array

    wvl_block : float
      wavelength window for each step
      (this parameter will be adjusted during the calculation)

    wvl_overlap : float
      the size of overlapping region
   
    output : str
      filename for the output 1d normalized spectrum
  
    '''

    wvl_range =np.max(long1d_wavelength)-np.min(long1d_wavelength) 
    if wvl_range < wvl_block:
      self.input_data(self,long1d_wavelength,long1d_flux,output=output)

    self.long1d_wavelength = long1d_wavelength
    self.long1d_flux = long1d_flux
    nblock = int((wvl_range - wvl_overlap) // (wvl_block-wvl_overlap)) + 1
    wvl_block = (wvl_range - wvl_overlap) / nblock + wvl_overlap
    ws = np.linspace(np.min(long1d_wavelength)-0.01,\
      np.max(long1d_wavelength) - wvl_overlap,
      nblock+1)[:-1]
    wf = ws + wvl_block
    wf[-1] = np.max(long1d_wavelength)+0.01 ## Make sure that the end point is equal to the maximum wavelength
    npt = np.array([np.sum((wws<=long1d_wavelength)&(long1d_wavelength<wwf)) for wws,wwf in zip(ws,wf)])
    ws = ws[npt > 0]
    wf = wf[npt > 0]
    multi_wavelength = [ \
      long1d_wavelength[(wws<=long1d_wavelength)&(long1d_wavelength<wwf)] \
      for wws,wwf in zip(ws,wf)\
      ]
    multi_flux = [ \
      long1d_flux[(wws<=long1d_wavelength)&(long1d_wavelength<wwf)] \
      for wws,wwf in zip(ws,wf)\
      ]
    self.n_overlap = [np.sum(multi_wavelength[ii]>=multi_wavelength[ii+1][0]) for ii in range(len(ws)-1)]
    self.input_multi_data(multi_wavelength,multi_flux,output_multi_head=None,output=None)
    if not output is None:
      self.output = output 

  ## Drawing functions
  def show_selected_region(self,samples):
    '''
    Display sampling regions. Also sort samples list. 

    Parameters
    ----------
    samples : list of list
      an example [[5000,5100],[5150,5250]]

    Returns
    -------
    samples sorted by wavelength
    '''

    self.canvas.samples = samples
    if hasattr(self.canvas,'vspan_samples'):
        _ = [vss.remove() for vss in getattr(self.canvas,'vspan_samples')]
    if len(samples)==0:
      self.canvas.vspan_samples = []
      self.canvas.draw()
      return samples
    else:
      ss = np.sort(samples)
      idx_ss = np.argsort(ss[:,0])
      vspan_samples = []
      x1,x2 = ss[idx_ss[0]]
      for idx in idx_ss[1:]:
        x1n,x2n = ss[idx]
        if x2 < x1n:
          vspan_samples.append(\
            self.canvas.axes.axvspan(x1,x2,facecolor='yellow',alpha=0.3,))
          x1 = x1n
          x2 = x2n
        elif x2n<x2:
          continue
        else:
          x2 = x2n
          continue
      vspan_samples.append(\
        self.canvas.axes.axvspan(x1,x2,facecolor='yellow',alpha=0.3,))
      self.canvas.vspan_samples = vspan_samples
      self.canvas.draw()
      return ss[idx_ss].tolist() 

  def draw_fig(self):
    '''
    Updates the figure
    '''
    assert hasattr(self,'wavelength'),'input_data first!'

    self.canvas.line_obs.set_xdata(self.CFit.wavelength)
    self.canvas.line_obs.set_ydata(self.CFit.flux)
    if hasattr(self.CFit,'continuum'):
      self.canvas.pt_use.set_xdata(self.CFit.wavelength[self.CFit.use_flag])
      self.canvas.pt_use.set_ydata(self.CFit.flux[self.CFit.use_flag])
      self.canvas.line_cont.set_xdata(self.CFit.wavelength)
      self.canvas.line_cont.set_ydata(self.CFit.flx_continuum)
      self.canvas.pt_knots.set_xdata(self.CFit.knotsx)
      self.canvas.pt_knots.set_ydata(self.CFit.knotsy)
    _ = self.show_selected_region(self.CFit.samples)
    self.canvas.draw()



# For future implementation
#  def on_pick(self,artist,mouseevent):
#    if self.mpl_status == 'u':
#      if (mouseevent.button == MouseButton.LEFT):
#        if hasattr(self.canvas,'vspan_samples'):
#          if (artist in self.canvas.vspan_samples):
#            xy = artist.get_xy()
#            xx = np.unique(xy[:,0])
#            ssarr = np.array(self.CFit.samples)
#            isremove = np.nonzero(\
#              (np.abs(np.round(ssarr[:,0]-np.min(xx),3))==0.000)|\
#              (np.abs(np.round(ssarr[:,1]-np.max(xx),3))==0.000))[0]
#            print(xx)
#            print(ssarr)
#            print(isremove)
#            ssremove = [self.CFit.samples[isr] for isr in isremove]
#            _ = [self.CFit.samples.remove(ssr) for ssr in ssremove]
#
#            self.CFit.samples = self.show_selected_region(self.CFit.samples)
#            self.ui.edit_samples.setPlainText(\
#              textsamples(self.CFit.samples))
#            self._clear_state()

  def _clear_state(self):
    '''
    This function is to clear a temporary state
    '''
    if hasattr(self,'tmp_data'):
      delattr(self,'tmp_data')
    self.canvas.mode_txt.set_text('Normal')
    self.mpl_status = None
    self.canvas.draw()

  def on_press(self,event):
    '''
    Key press event

    FUTURE IMPLEMENTATION
    - custom key bindings
    - help
    '''

    print(event.key)
    if self.mpl_status is None:
      if (event.key=='s') and (not event.xdata is None):
        self.mpl_status = 's'
        self.tmp_data = {'x':event.xdata,
          'lvx1':self.canvas.axes.axvline(event.xdata,color='r',lw=2.)}
        self.canvas.mode_txt.set_text('Sample')
        self.canvas.draw()
      elif event.key == 'u':
        ### Future implementation of removing a data point
        pass
        #self.mpl_status = 'u'
        #self.canvas.mode_txt.set_text('Un-sample')
        #self.canvas.draw()
      elif event.key == 't':
        print('Clear samples \n')
        self.CFit.samples = []
        self.CFit.samples = self.show_selected_region(self.CFit.samples)
        self.ui.edit_samples.setPlainText(\
          utils.textsamples(self.CFit.samples))
      elif event.key == 'n':
        self.moveon_done()
      elif event.key == 'f':
        print('Refit')
        self.CFit.continuum(self.wavelength,self.flux)
        self.draw_fig()        
    elif self.mpl_status == 's':
      if (event.key=='s') and (not event.xdata is None):
        x1 = self.tmp_data['x']
        x2 = event.xdata
        self.CFit.samples.append([x1,x2])
        self.CFit.samples = self.show_selected_region(self.CFit.samples)
        self.ui.edit_samples.setPlainText(\
          utils.textsamples(self.CFit.samples))
      self.tmp_data['lvx1'].remove()
      self._clear_state()

  def eventFilter(self,source,event):
    '''
    Reflect text box edits 

    FUTURE IMPLEMENTATION
    - better implementation? currently a bit redundant
    '''
    if event.type() == QEvent.FocusIn:
      if source is self.ui.edit_samples:
        self.temp_text = source.toPlainText()
      else:
        self.temp_text = source.text()
    elif event.type() == QEvent.FocusOut:
      if source is self.ui.edit_samples:
        new_text = source.toPlainText()
      else:
        new_text = source.text()
      if source is self.ui.edit_function:
        if new_text in ['spline3']:
          self.CFit.func = new_text
        else:
          self.ui.edit_function.setText(self.temp_text)
      elif source is self.ui.edit_knots:
        try:
          self.CFit.dwvl_knots = float(new_text)
        except:
          self.ui.edit_knots.setText(self.temp_text)
      elif source is self.ui.edit_lowrej:
        try:
          self.CFit.low_rej = float(new_text)
        except:
          self.ui.edit_lowrej.setText(self.temp_text)
      elif source is self.ui.edit_highrej:
        try:
          self.CFit.high_rej = float(new_text)
        except:
          self.ui.edit_highrej.setText(self.temp_text)
      elif source is self.ui.edit_grow:
        try:
          self.CFit.grow = float(new_text)
        except:
          self.ui.edit_grow.setText(self.temp_text)
      elif source is self.ui.edit_nave:
        try:
          self.CFit.naverage = round(float(new_text))
        except:
          self.ui.edit_nave.setText(self.temp_text)
      elif source is self.ui.edit_niter:
        try:
          self.CFit.niterate = round(float((new_text)))
        except:
          self.ui.edit_niter.setText(self.temp_text)
      elif source is self.ui.edit_samples:
        try:
          ss = utils.textsamples(new_text,reverse=True)
          ss_sorted = self.show_selected_region(ss)
          self.CFit.samples = ss_sorted
          self.ui.edit_samples.setPlainText(\
            utils.textsamples(ss_sorted))
        except:
          print('Input error')
          self.ui.edit_samples.setPlainText(self.temp_text)
    elif event.type() == QEvent.MouseButtonPress:
      if source is self.ui.button_fit:
        print('Refit')
        self.CFit.continuum(self.wavelength,self.flux)
        self.draw_fig()
      elif source is self.ui.button_draw:
        self.draw_fig()
    return QWidget.eventFilter(self, source, event)

  def _sum_2spec(self,x1,y1,x2,y2):
    '''
    Sums two spectra that have an overlapping region

    Parameters
    ----------
    x1, x2: 1d-array
      wavelength of the spectra

    y1, y2: 1d-or 2d-array
      flux of the spectra
    '''

    try:
      len(y1[0])
      pass
    except:
      y1 = [y1]
      y2 = [y2]
    if x1[0] > x2[0]:
      x1,x2 = x2,x1
      y1,y2 = y2,y1
    if x1[-1]>x2[0]: # Overlap
        j1 = np.nonzero(x1>x2[0])[0][0]-1
        j2 = np.nonzero(x2<x1[-1])[0][-1]+1
        x_mid = np.linspace(x1[j1],x2[j2],
          int(np.maximum(len(x1)-j1,j2+1)))
        y_mid = [\
          utils.rebin(x1[j1:],yy1[j1:],x_mid,conserve_count=True)+\
          utils.rebin(x2[:j2+1],yy2[:j2+1],x_mid,conserve_count=True) 
          for yy1,yy2 in zip(y1,y2)]

        xout = np.append(np.append(\
          x1[:j1],x_mid),
          x2[j2+1:])
        yout = [np.append(np.append(\
          yy1[:j1],yy_mid),
          yy2[j2+1:]) \
          for yy1,yy2,yy_mid in zip(y1,y2,y_mid)]
        return (xout,)+tuple(yout)
    else: # No overlap
      xout = np.append(x1,x2)
      yout = [np.append(yy1,yy2) for yy1,yy2 in zip(y1,y2)]
      return (xout,)+tuple(yout)

  def done(self):
    '''
    Displays a message that all the fitting are completed.
    '''
    self.canvas.axes.text(0.5,0.5,'Done! \n Close the window',
      bbox=dict(facecolor='white', alpha=0.5),
      transform=self.canvas.fig.transFigure,
      fontsize='xx-large',color='k',
      horizontalalignment='center',verticalalignment='center')

  def multi_done(self,output):
    '''
    Completes the analysis and outputs results in case the input
    spectral format is multispec.
    Three output files are createad: 1d normalized, blaze, 1d combined
    
    Parameters
    ----------
    output : str
      the filename of the output file
    '''
    wvl1d,flx1d,blaze1d = utils.x_sorted(
      self.multi_wavelength[0],
      [self.multi_flux[0],
      self.multi_blaze[0]])
    for ii in range(1,self.norder):
      wvl_ii,flx_ii,blaze_ii = utils.x_sorted(
        self.multi_wavelength[ii],
        [self.multi_flux[ii],
        self.multi_blaze[ii]])
      wvl1d,flx1d,blaze1d = self._sum_2spec(\
        wvl1d,[flx1d,blaze1d],wvl_ii,[flx_ii,blaze_ii])
    print('Continuum fitting for a spectrum in multispec format was completed')
    print(f'The resut is saved as {output}')
    np.savetxt(self.output,\
      np.array([wvl1d,flx1d/blaze1d]).T,fmt='%12.6f')
    np.savetxt(os.path.dirname(self.output)+'/blaze_'+os.path.basename(self.output),\
      np.array([wvl1d,blaze1d]).T,fmt='%12.6f')
    np.savetxt(os.path.dirname(self.output)+'/1d_'+os.path.basename(self.output),\
      np.array([wvl1d,flx1d]).T,fmt='%12.6f')

  def long1d_done(self,output):
    '''
    Completes the analysis and outputs results in case the input
    spectral format is long1d.
    
    Parameters
    ----------
    output : str
      the filename of the output file
    '''
    fweight = lambda n: np.where(np.arange(0,n)<(n/4),0.0,\
      np.where(np.arange(0,n)>(3.0*n/4),1.0,\
      (np.arange(0,n)-n/4)*2/n))
      
    nblock = len(self.multi_wavelength)
    blaze1d = np.zeros(len(self.long1d_wavelength))
    n1,n2 = 0,0
    nn = (0,0)
    for ii in range(nblock):
      n1 = n2+nn[1]
      n2 = n1+len(self.multi_wavelength[ii])
      if ii == 0:
        nn = (0,-self.n_overlap[ii])
      elif ii+1 == nblock:
        nn = (self.n_overlap[ii-1],-0)
      else:
        nn = (self.n_overlap[ii-1],-self.n_overlap[ii])
      ## For no overlapping region
      blaze1d[n1+nn[0]:n2+nn[1]] = \
        self.multi_blaze[ii][nn[0]:len(self.multi_wavelength[ii])+nn[1]]
      if ii != 0:      ## For region overlapping with previous section
        blaze1d[n1:n1+nn[0]] = \
          self.multi_blaze[ii][0:nn[0]]*fweight(nn[0])
      if ii+1 != nblock: ## For region overlapping with next section
        blaze1d[n2+nn[1]:n2] = \
          self.multi_blaze[ii][len(self.multi_wavelength[ii])+nn[1]:]*\
              (1.0-fweight(np.abs(nn[1])))
    self.long1d_normalized = self.long1d_flux / blaze1d
    print('Continuum fitting for a spectrum in long1d format was completed')
    print(f'The resut is saved as {output}')
    np.savetxt(output,
      np.array([self.long1d_wavelength,\
                self.long1d_normalized]).T,fmt='%12.6f')

  def moveon_done(self):
    '''
    Moves on to next step /order after completing comtinuum fitting for 
    one section.
    '''
    if len(self.wavelength)!=len(self.CFit.wavelength):
      self.blaze = np.interp(self.wavelength,
        self.CFit.wavelength,
        self.CFit.flx_continuum)
    else:
      self.blaze = self.CFit.flx_continuum
    self.normalized = self.flux/self.blaze
    if hasattr(self,'multi_wavelength'):
      self.multi_blaze.append(self.blaze)        
      self.multi_normalized.append(self.normalized)
      if hasattr(self ,'output_multi_head'):
        np.savetxt(self.output_multi_head+\
          '{0:03d}details.csv'.format(self.current_order),
        np.array([self.wavelength,\
          self.flux,\
          self.blaze,\
          self.normalized]).T,fmt='%12.6f')
      self.current_order +=1
      if self.current_order == self.norder:
        if hasattr(self,'output'):
          if hasattr(self,'long1d_wavelength'):
            self.long1d_done(self.output)
          else:
            self.multi_done(self.output)
        self.done()
      else:
        self.CFit.samples = []
        self.CFit.samples = self.show_selected_region(self.CFit.samples)
        self.ui.edit_samples.setPlainText(\
          utils.textsamples(self.CFit.samples))
        self.input_data(\
          self.multi_wavelength[self.current_order],
          self.multi_flux[self.current_order],)
    else:
      if hasattr(self,'output'):
        np.savetxt(self.output,
          np.array([self.wavelength,\
                    self.normalized]).T,fmt='%12.6f')
      self.done()



def start_gui(wavelength,flux,outfile,form='1d',output_multi_head=None):
  '''
  This function starts the gui. 

  Parameters
  ----------
  wavelength : array
  
  flux : 1d- or 2d-array
  
  outfile : str
    the name of the output normalized spectrum 
  
  form : str
    supported formats are 
    - 1d
      The flux is just normalized
    - long1d
      This option should be used when the input spectrum is 1d and covers 
      a wide wavelength ragnge. The input spectrum will be divided into small 
      pieces because of computational reasons. There are small overlapping 
      regions so that there are no discontinuities 
    - multi
      This option is used when the input spectrum has a multispec format.
      Continuum placement is done for each order. The normalized spectrum
      is combined to form an 1d normalized spectrum for the output.
  
  output_multi_head : str
    Used only when form is 'multi'. Blaze function and normalized spectra
    are written to a file for each order. The filenames of these spectra 
    start with the string specified by this parameter.      

  '''

  app = QApplication(sys.argv)
  window = MainWindow()
  if form == 'long1d':
     window.input_long1d(\
       wavelength,flux,\
       output=outfile)
  elif form == 'multi':
     window.input_multi_data(\
       wavelength,flux,\
       output=outfile,\
       output_multi_head=output_multi_head) 
  else:
     window.input_data(\
       wavelength,flux,\
       output=outfile)
  window.show()
  sys.exit(app.exec_())


