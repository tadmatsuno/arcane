import pandas
from solarabundance21 import getamass
import numpy as np
from astropy.constants import k_B,a0
import astropy.units as u
from scipy.special import gamma

class Linelist(pandas.DataFrame):
    '''
    Class to store a linelist. It currently supports VALD formats
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def read_vald(filename):
    with open(filename, 'r') as f:
        line = f.readline()
        while (line.lstrip().startswith('#')):
            # Read until the header, which should contain 'Ion'
            line = f.readline()
        extract_stellar = 'lines selected,' in line # Linelist extracted from stellar has ''lines selected,' in the header
        while (not 'Ion' in line):
            # Read until the header, which should contain 'Ion'
            line = f.readline()
        isshort = 'Reference' in line
    if isshort:
        linelist = read_valdshort(filename)
    else:
        linelist = read_valdlong(filename)
    return linelist

def convert_sigma_alpha_to_gamma(elem_num, sigma, alpha,t0=1.0e4*u.K):
    elemnum = int(elem_num)
    reduced_mass = 1.0 / \
        ( 1.0/getamass(1,pnum='yes') + 1.0/getamass(elemnum,pnum='yes'))
    vbar = np.sqrt(8.0*k_B*t0 / (np.pi*reduced_mass*u.u))
    v0 = 1.0e4*u.m/u.s
    sigma = sigma*a0**2.
    ww = (4.0/np.pi)**(alpha/2.)*gamma((4.0-alpha)/2.0)*vbar*sigma*((vbar/v0).decompose())**(-alpha)
    return np.log10((2.*ww/(u.cm**3/u.s)).decompose())


def read_valdshort(filename):
    linelist = []
    with open(filename, 'r') as f:
        line = f.readline()
        while (line.lstrip().startswith('#')):
            # Read until the header, which should contain 'Ion'
            line = f.readline()
        extract_stellar = 'lines selected,' in line # Linelist extracted from stellar has ''lines selected,' in the header
        while (not 'Ion' in line):
            # Read until the header, which should contain 'Ion'
            line = f.readline()
        line = f.readline()
        while (line.count(',')>5):
            values = line.split(',')
            if extract_stellar:
                species = values[0].replace("'","")
                wavelength = float(values[1])
                expot = float(values[2])
                loggf = float(values[4])
                gamrad = float(values[5])
                gamqst = float(values[6])
                gamvw = float(values[7])
                lande = float(values[8])
                depth = float(values[9])
                references = values[10]
                hfs = 'hfs' in references
                linelist.append(\
                    [species, wavelength, expot, loggf, gamrad, gamqst, gamvw, lande, depth,references,hfs])
            else:
                species = values[0].replace("'","")
                wavelength = float(values[1])
                expot = float(values[2])
                loggf = float(values[3])
                gamrad = float(values[4])
                gamqst = float(values[5])
                gamvw = float(values[6])
                lande = float(values[7])
                references = values[8]
                hfs = 'hfs' in references
                linelist.append(\
                    [species, wavelength, expot, loggf, gamrad, gamqst, gamvw, lande, references,hfs])
            line = f.readline()
        if extract_stellar:
            linelist = Linelist(linelist, \
                columns=['species', 'wavelength', 'expot', 'loggf',\
                    'gamma_rad', 'gamma_stark', 'gamma_vw', 'lande', 'depth', 'references','HFS'])
        else:
            linelist = Linelist(linelist, \
                columns=['species', 'wavelength', 'expot', 'loggf',\
                'gamma_rad', 'gamma_stark', 'gamma_vw', 'lande', 'references','HFS'])
    return linelist

def read_valdlong(filename):
    linelist = []
    with open(filename, 'r') as f:
        line = f.readline()
        while (line.lstrip().startswith('#')):
            # Read until the header, which should contain 'Ion'
            line = f.readline()
        extract_stellar = 'lines selected,' in line # Linelist extracted from stellar has ''lines selected,' in the header
        while (not 'Ion' in line):
            # Read until the header, which should contain 'Ion'
            line = f.readline()
        line = f.readline()
        while (line.count(',')>5):
            values = line.split(',')
            if extract_stellar:
                species = values[0].replace("'","")
                wavelength = float(values[1])
                loggf = float(values[2])
                expot_lo = float(values[3])
                j_lo = float(values[4])
                expot_up = float(values[5])
                j_up = float(values[6])
                lande_lo = float(values[7])
                lande_up = float(values[8])
                lande = float(values[9])
                gamrad = float(values[10])
                gamqst = float(values[11])
                gamvw = float(values[12])
                depth = float(values[13])
            else:
                species = values[0].replace("'","")
                wavelength = float(values[1])
                loggf = float(values[2])
                expot_lo = float(values[3])
                j_lo = float(values[4])
                expot_up = float(values[5])
                j_up = float(values[6])
                lande_lo = float(values[7])
                lande_up = float(values[8])
                lande = float(values[9])
                gamrad = float(values[10])
                gamqst = float(values[11])
                gamvw = float(values[12])
            #read configurations
            def get_configurations(line):
                if (len(line) - line.count(' '))>0:
                    coupling = line[0:5].replace(' ','')
                    line_rest = line[5:].lstrip().split()
                    if len(line_rest)>1:
                        electron_conf = line_rest[0]
                        term_desig = line_rest[1]
                    elif len(line_rest)==1:
                        electron_conf = line_rest[0]
                        term_desig = ''
                    else:
                        electron_conf = ''
                        term_desig = ''
                return coupling, electron_conf, term_desig
            line = f.readline()
            coupling_lo, electron_conf_lo, term_desig_lo = get_configurations(line)
            line = f.readline()
            coupling_up, electron_conf_up, term_desig_up = get_configurations(line)
            # Reference line
            line = f.readline()
            references = line
            hfs = 'hfs' in references
            if extract_stellar:
                linelist.append(\
                    [species, wavelength, loggf, expot_lo, j_lo, expot_up, j_up, \
                    lande_lo, lande_up, lande, gamrad, gamqst, gamvw, depth, \
                    coupling_lo, electron_conf_lo,  term_desig_lo, \
                    coupling_up, electron_conf_up, term_desig_up, references, hfs])
            else:
                linelist.append(\
                    [species, wavelength, loggf, expot_lo, j_lo, expot_up, j_up, \
                    lande_lo, lande_up, lande, gamrad, gamqst, gamvw, \
                    coupling_lo, electron_conf_lo,  term_desig_lo, \
                    coupling_up, electron_conf_up, term_desig_up, references, hfs])
            line = f.readline()
        if extract_stellar:
            linelist = Linelist(linelist, \
                columns=['species', 'wavelength', 'loggf', 'expot', 'j_lo', 'expot_up', 'j_up', \
                    'lande_lo', 'lande_up', 'lande', 'gamma_rad', 'gamma_stark', 'gamma_vw', 'depth', \
                    'coupling_lo', 'electron_conf_lo', 'term_desig_lo',
                    'coupling_up', 'electron_conf_up', 'term_desig_hi', 'references', 'HFS'])
        else:
            linelist = Linelist(linelist, \
                columns=['species', 'wavelength', 'loggf', 'expot', 'j_lo', 'expot_up', 'j_up', \
                    'lande_lo', 'lande_up', 'lande', 'gamma_rad', 'gamma_stark', 'gamma_vw', \
                    'coupling_lo', 'electron_conf_lo', 'term_desig_lo',
                    'coupling_up', 'electron_conf_up', 'term_desig_hi', 'references', 'HFS'])
    return linelist
