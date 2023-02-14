import shutil
from functools import wraps
import os
import shutil
import pandas
import warnings
import numpy as np
from arcane_dev.utils.solarabundance import get_atomnum
from arcane_dev.mdlatm import marcs

## Detail see Params.f
moog_default_input = {
    'freeform' : 0,
    'atmosphere':1,
    'molecules':2,
    'lines':1,
    'gfstyle':0,
    'units':0,
    'flux/int':0,
    'damping':1,
    'scat':1,
    'molset':1,
    'plot':0
}

solar_moog = [
    12.00, 10.93, 1.05, 1.38, 2.70, 8.43, 7.83, 8.69, 4.56, 7.93,
    6.24, 7.60, 6.45, 7.51, 5.41, 7.12, 5.50, 6.40, 5.03, 6.34,
    3.15, 4.95, 3.93, 5.64, 5.43, 7.50, 4.99, 6.22, 4.19, 4.56,
    3.04, 3.65, 2.30, 3.34, 2.54, 3.25, 2.52, 2.87, 2.21, 2.58,
    1.46, 1.88, -5.00, 1.75, 0.91, 1.57, 0.94, 1.71, 0.80, 2.04,
    1.01, 2.18, 1.55, 2.24, 1.08, 2.18, 1.10, 1.58, 0.72, 1.42,
    -5.00, 0.96, 0.52, 1.07, 0.30, 1.10, 0.48, 0.92, 0.10, 0.84,
    0.10, 0.85, -0.12, 0.85, 0.26, 1.40, 1.38, 1.62, 0.92, 1.17,
    0.90, 1.75, 0.65, -5.00, -5.00, -5.00, -5.00, -5.00, -5.00, 0.02,
    -5.00, -0.54, -5.00, -5.00, -5.00
]


def get_moog_species_id(species):
    '''
    MgI -> 12.0
    Mg1 -> 12.0
    CH -> 106.00000
    caveat 1: can't deal with molecules ending with I
    caveat 2: doesn't support isotopes
    '''
    species = species.replace(' ','')
    if (species[-1]!='I') and (not species[-1].isnumeric()):
        # molecule
        atomnums = []
        i0 = 0
        for ii,s in enumerate(species):
            if s.isupper():
                if ii!=0:
                    atomnums.append(get_atomnum(species[i0,ii+1]))
                i0 = ii
        atomnums = np.sort(atomnums)[::-1]
        species_id = '{0:d}.00000'.format(\
            np.sum([atomnums[i]*10**(2*i) for  i in range(len(atomnums))]))
    else:
        # atom
        if species[-1].isnumeric():
            species_id = '{0:2d}.{1:1d}'.format(\
                get_atomnum(species[:-1]),(int(species[-1])-1))
        else:
            species_id = '{0:2d}.{1:1d}'.format(\
                get_atomnum(species[:species.find('I',1)]),
                species.count('I',1)-1)
    return species_id

def write_marcs2moog_model(fname,model,vt,feh_overwrite = None):
    ### MOOG treats all the models as plane-parallel (Right?).
    ### This has an inconsistency if you use a spherical model for giant,
    ### but spherical models are better than plane-parallel models even if 
    ### the radiative transfer code assumes plane-parallel (Heiter & Eriksson 2006)
    if type(model) is str:
        model = marcs.read_marcs(model)
    with open(fname,'w') as f:
        f.write('BEGN\n')
        f.write('MARCS {0:1s}_t{1:4.0f}g{2:5.2f}m{3:5.2f}a{4:5.2f}\n'.format(\
            model['geometry'][0],model['teff'],model['logg'],model['m_h'],model['alpha_m']))
        f.write('{0:10s}{1:d}\n'.format('NTAU',model['ndepth']))
        for ii in range(model['ndepth']):
            f.write('{0:10.3f} {1:10.1f} {2:.4E} {3:.4E} {4:.4f} {5:.4E}\n'.format(\
                model['lgTauR'][ii],
                model['T'][ii],
                model['Pg'][ii],
                model['Pe'][ii],
                model['Mu'][ii],
                model['KappaRoss'][ii],
                ))
        f.write(f'{vt:6.3f}\n')
        if feh_overwrite is None:
            f.write('{0:10s}{1:3d}{2:8.3f}\n'.format('NATOM',0,model['m_h']))
        else:
            f.write('{0:10s}{1:3d}{2:8.3f}\n'.format('NATOM',0,feh_overwrite))
 
        ###TODO ### 
        ### I think it should allow to have a metallicity that is different from what is used 
        ### to construct a model atmosphere. UMP/HMP stars might not have a corresponding marcs
        ### model computed.
        ### Sholud I also allow turbospectrum like abundance input? alpha/Fe, s/Fe, r/Fe etc?

        f.write('{0:10s}{1:d}\n'.format('NMOL',22))
        f.write('  101   106   107   108   112  126\n') 
        f.write('  606   607   608\n')                  
        f.write('  707   708\n')                        
        f.write('  808   812   822\n')                  
        f.write('  10108 60808\n')                      
        f.write('  1.1 6.1     7.1     8.1   12.1  22.1  26.1\n') 

def run_moog(mode, linelist, run_id = '', workdir = '.',
    moog_mod_file = None, marcs_mod_file = None, 
    teff = None, logg = None, feh = None, alphafe = None, 
    feh_mod = None, alphafe_mod = None,
    vt = None, 
    defalut_dampnum = 3.,
    species_vary = 0,
    dwvl_margin = 2.0,
    dwvl_step = 0.01,
    cog_ew_minmax = [-7,-4],
    **kw_args):

    flinelist = f'{workdir}/line_{run_id}.in'
    fmodelin  = f'{workdir}/model_{run_id}.in'
    fstdout   = f'{workdir}/moog_{run_id}.std'
    fsummary  = f'{workdir}/moog_{run_id}.sum'
    flog  = f'{workdir}/moog_{run_id}.log'


    if (mode in ['cog','cogsyn','blends']) and (species_vary ==0):
        raise ValueError(f'For {key} driver, species_vary needs to be specified')
    abundances = {}
    isotopes = {}
    for key,val in kw_args.items():
        if key[0] == 'A':
            atomnum = int(key[2:])
            abundances[atomnum] = val - (solar_moog[atomnum-1] + feh)
        if key[0] == 'I':
            isospecies = '.'.join(key[2:].split('_'))
            isotopes[isospecies] = 1.0/val

    if (type(linelist) is dict) or (type(linelist) is pandas.DataFrame): 
        # Create linelist if it is not a filename
        
        if type(linelist) is pandas.DataFrame: 
            # Convert pandas DataFrame to dict
            linelist = linelist.to_dict(orient='list')
        # First check if linelist dict has all the necessary information
        for key in ['wavelength','chi','loggf']:
            assert key in linelist.keys(), f'{key} needs to be in linelist keys'
        assert ('species' in linelist.keys())|('moog_species' in linelist.keys()),\
            'species or moog_species needs to be in linelist keys'

        if not 'moog_species' in linelist.keys():
            linelist['moog_species'] = get_moog_species_id(linelist['species'])
        nline = len(linelist['wavelength'])
        # Fill missing information with 0
        if not 'ew' in linelist.keys():
            if mode == 'abfind':
                warnings.warn('you are supposed to provide ew for the task abfind')
            linelist['ew'] = np.zeros(nline)
        linelist['ew'] = np.where(np.isfinite(linelist['ew']),linelist['ew'],0.0)
        if not 'dampnum' in linelist.keys():
            linelist['dampnum'] = np.zeros(nline) + defalut_dampnum
        linelist['dampnum'] = np.where(np.isfinite(linelist['dampnum']),
                linelist['dampnum'],defalut_dampnum)
        if not 'd0' in linelist.keys():
            linelist['d0'] = np.zeros(nline)
        linelist['d0'] = np.where(np.isfinite(linelist['d0']),linelist['d0'],0.0)

        # Write linelist
        with open(flinelist, 'w') as f:
            f.write('Linelist createad by arcane\n')
            #TODO:Support for HFS?   
            for ii in range(nline):
                if linelist['chi'][ii] >= 50:
                    warnings.warn('lines with chi>50 ev will be skipped')
                    continue
                f.write("{0:10.3f}{1:>10s}{2:10.3f}{3:10.3f}{4:10.3f}{5:10.3f}{6:10.3f}\n".\
                    format(linelist['wavelength'][ii],
                        linelist['moog_species'][ii],
                        linelist['chi'][ii],
                        linelist['loggf'][ii],
                        linelist['dampnum'][ii],
                        linelist['d0'][ii],
                        linelist['ew'][ii]
                ))
        wmin,wmax = np.min(linelist['wavelength']),np.max(linelist['wavelength'])
    else:
        shutil.copy(linelist, flinelist)
        with open(flinelist,'r') as f:
            ww = [float(line.split()[0]) for line in f.readlines()[1:]]
        wmin,wmax = np.min(ww),np.max(ww)

    if not moog_mod_file is None:
        assert os.path.exists(moog_mod_file), 'moog_mod_file specified does not exist'
        if any([not val is None for val in \
            [marcs_mod_file, teff, logg, feh, alphafe, feh_mod, alphafe_mod, vt]]):
            warnings.warn('moog_mod_file is provided. '+\
                'marc_mod_file, teff, logg, feh, alphafe, feh_mod, alphafe_mod will be ignored')
        shutil.copy(moog_mod_file,fmodelin)
    elif not marcs_mod_file is None:
        assert os.path.exists(marcs_mod_file),'marcs_mod_file specified does not exist'
        if not all((teff is None, logg is None,feh_mod is None, alphafe_mod is None )):
            warnings.warn(
                'MARCS model file is directly provided. '+\
                'Teff, logg, [Fe/H]_mod, and [alpha/Fe]_mod will be ignored')
        assert not vt is None,'vt is needed'
        model = marcs.read_marcs(marcs_mod_file)
        write_marcs2moog_model(fmodelin,model,vt, feh_overwrite = feh)
    else:
        if feh_mod is None:
            feh_mod = feh
        if alphafe_mod is None:
            alphafe_mod = alphafe
        if any([val is None for val in [teff, logg, feh_mod, vt]]):
            raise ValueError('At least four parameters, teff, logg, feh_mod,vt are needed')
        modelatm_file = f'{workdir}/marcs_{run_id}.mod'
        model = marcs.get_marcs_mod(teff, logg, feh_mod, \
            alphafe=alphafe_mod, outofgrid_error=True)
        write_marcs2moog_model(fmodelin,model,vt, feh_overwrite = feh)
 
    
    ## Write MOOG input files
    if key in kw_args.keys():
        if (not (key[0] in ['I','A'])) and (not (key in moog_default_input.keys())):
            warnings.warn(f'{key} is not in moog default input list,'+
                'and thus will be ignored')
    with open('batch.par','w') as f:
        f.write(mode+'\n')
        f.write('{0:20s}"{1:s}"\n'.format('standard_out',fstdout))
        f.write('{0:20s}"{1:s}"\n'.format('summary_out',fsummary))
        f.write('{0:20s}"{1:s}"\n'.format('model_in',fmodelin))
        f.write('{0:20s}"{1:s}"\n'.format('lines_in',flinelist))
        for key,val in moog_default_input.items():
            if key in kw_args.keys():
                val = kw_args[key] # Overwrite default parameter 
            f.write('{0:20s}{1:d}\n'.format(key,val))
        if len(abundances)>0:
            f.write('{0:20s}{1:d} 1\n'.format('abundances',len(abundances)))
            for key,val in abundances.items():
                f.write('{0:d} {1:7.3f}\n'.format(key,val))
        if len(isotopes)>0:
            f.write('{0:20s}{1:d} 1\n'.format('isotopes',len(isotopes)))
            for key,val in isotopes.items():
                f.write('{0:s} {1:7.3f}\n'.format(key,val))
        if mode in ['synth']:
            f.write('synlimits\n')
            f.write('{0:10.3f}{1:10.3f}{2:10.3f}{3:10.3f}\n'.format(\
                wmin-dwvl_margin,wmax+dwvl_margin,dwvl_step,dwvl_margin))
        if mode in ['cog','cogsyn']:
            f.write('coglimits\n')
            f.write('{0:10.3f}{1:10.3f}{2:10.3f}{3:10.3f}{4:5d}\n'.format(\
                cog_ew_minmax[0],cog_ew_minmax[1],0.1,species_vary))
        if mode in ['blends']:
            f.write('blenlimits\n')
            f.write('{0:10.3f}{1:10.3f}{2:10.3f}\n'.format(\
                dwvl_margin,dwvl_step,species_vary))
        

    ## Run MOOG
    os.system(f'MOOGSILENT > {flog} 2>&1')
    
    return fsummary

        
def synth(**kwargs):
    '''
    Usage:

    1. to specify abundance of an element, provide A_{proton_number} = log(N_X/N_H) + 12
        e.g., A_6 = 8.43
    2. to specify isotope ratio, provide I_{proton number}_{atomic_mass}
        e.g., I_6_13 = 0.01
    '''
    fsummary = run_moog('synth',**kwargs)
    with open(fsummary,'r') as f:
        line = ''
        while not line.startswith('MODEL'):
            line = f.readline()
        line = f.readline()
        ws,wf,dwvl,wm = [float(l) for l in line.strip().split()]
        wvl = np.arange(ws,wf+dwvl,dwvl)
        flux = []
        for line in f.readlines():
            for ii in range(10):
                l = line[ii*7:(ii+1)*7]
                if l.strip() != '':
                    flux.append(float(l))
    flux = np.array(flux)
    wvl = wvl[0:len(flux)]
    return wvl,1.0-flux


