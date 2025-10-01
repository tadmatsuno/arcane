from arcane.synthesis import moog
from arcane.spectrum import model
import unittest
import os
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from arcane.synthesis import readvald

cmap = cm.get_cmap('viridis')

linelist = pandas.read_csv('./vald/stellar_short.csv',dtype={'moog_species':str})
linelist2 = readvald.read_valdshort('./vald//Vald_all_short')

def make_plot_of_grid(linelist):
    input_dict = {'teff':4636,'logg':1.418,'vt':2.05,'feh':-2.60,'A_6':5.220,'A_56':-1.80,
        'linelist':linelist}
    absorption = model.LineSynth1param(\
        moog.synth,
        input_dict,
        'A_56',
        vfwhm_in = 8.0,
        grid_size = 0.1, 
        grid_scale = 'linear', 
        snr=30.,
        niterate = 2, 
        low_rej = 3., high_rej = 3., grow = 0.05,
        naverage = 1, fit_mode = 'subtract',
        samples = [[5850,5860]], 
        std_from_central = False, 
        kw_fit_control = {'fix_vFWHM':True})
    absorption.construct_grid()

    fig,ax = plt.subplots()
    
    norm = Normalize(vmin=-2-1.8,vmax=2-1.8)
    for key in absorption.grid.keys():
        if isinstance(key,int):
            ax.plot(absorption.grid['wvl'],1.0-absorption.grid[key],color=cmap(norm(key*absorption.grid_size)))
    cax = fig.add_axes([0.85, 0.2, 0.02, 0.6])  
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Z values')

    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Flux')
    ax.set_xlim(5853,5855)
    fig.tight_layout()
    fig.savefig('img/test_construct_grid.png')
    return absorption



if __name__ == '__main__':
    make_plot_of_grid(linelist2)