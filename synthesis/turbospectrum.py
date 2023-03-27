import shutil
import os
import warnings
from arcane_dev.mdlatm import marcs

ts_default_input = {
    'INTENSITY_FLUX':'Flux',
    'R_PROCESS':0.0,
    'S_PROCESS':0.0,
    'ALPHA_Fe' :0.0,
    'HELIUM' : 0.0,
    'ABFIND' : True,
}

def run_turbospectrum(mode,
    linelist = None, run_id='',workdir='.',
    ts_opac_file = None, marcs_mod_file = None,
    teff = None, logg = None, feh = None, alphafe = None, 
    feh_mod = None, alphafe_mod = None,
    vt = None,
    defalut_dampnum = 3.,
    wvl_minmax =None,
    dwvl_margin = 2.0,
    dwvl_step = 0.01,
    spherical = False,
    **kw_args):
    '''
    Run turbospectrum. 
    
    See TURBOSPECTRUM/source/input.f for input parameters.

    Parameters
    ----------
    mode : str
        'babsma', 'syn', 'eqwidt'
    '''
    fscript = f'{workdir}/turbospectrum_{run_id}.sh'

    if (mode != 'babsma') and (linelist is None):
        raise ValueError('linelist is needed for mode other than babsma')
    
    # Need a part that reads or writes the linelist and get the wavelength range

    # Need a part where input are set up and kw_args are processed
    ts_input = ts_default_input.copy()
    abundances = {}
    isotopes = {}
    for key,val in kw_args.items():
        if key[0] == 'A':
            atomnum = int(key[2:])
            abundances[atomnum] = val
        if key[0] == 'I':
            isospecies = '.'.join(key[2:].split('_'))
            isotopes[isospecies] = val

    if (mode != 'babsma') and (ts_opac_file is None):
        ts_opac_file = run_turbospectrum('babsma',
            run_id=run_id+'opac', workdir=workdir, 
            marcs_mod_file = marcs_mod_file, 
            teff = teff, logg = logg, feh = feh, alphafe = alphafe,
            feh_mod = feh_mod, alphafe_mod = alphafe_mod,
            vt = vt, defalut_dampnum= defalut_dampnum, 
            wvl_minmax = (wvlmin-5, wvlmax+5))
    if (mode == 'babsma'):
        if not marcs_mod_file is None:
            assert os.path.exists(marcs_mod_file),'marcs_mod_file specified does not exist'
            if not all((teff is None, logg is None,feh_mod is None, alphafe_mod is None )):
                warnings.warn(
                    'MARCS model file is directly provided. '+\
                    'Teff, logg, [Fe/H]_mod, and [alpha/Fe]_mod will be ignored')
            model = marcs.read_marcs(marcs_mod_file)
        else:
            if feh_mod is None:
                feh_mod = feh
            if alphafe_mod is None:
                alphafe_mod = alphafe
            if any([val is None for val in [teff, logg, feh_mod, vt]]):
                raise ValueError('At least four parameters, teff, logg, feh_mod,vt are needed')
            marcs_mod_file = f'{workdir}/marcs_{run_id}.mod'
            model = marcs.get_marcs_mod(teff, logg, feh_mod, \
                alphafe=alphafe_mod, outofgrid_error=True)
            marcs.write_marcs(model, marcs_mod_file)
        if feh is None:
            feh = model['m_h']
        if alphafe is None:
            alphafe = model['alpha_fe']
        fresult = f'{workdir}/turbospectrum_{run_id}.opac'
    else:
        if not marcs_mod_file is None:
            assert os.path.exists(marcs_mod_file),'marcs_mod_file specified does not exist'
            model = marcs.read_marcs(marcs_mod_file)['m_h']
            if feh_mod is None:
                feh_mod = model['m_h']
            if alphafe_mod is None:
                alphafe_mod = model['alpha_fe']
        if feh is None:
            feh = feh_mod
            if feh is None:
                raise ValueError('feh is not specified')
        if alphafe is None:
            alphafe = alphafe_mod
        fresult = f'{workdir}/turbospectrum_{run_id}.out'
    if not alphafe is None:
        ts_input['ALPHA_FE'] = alphafe

    f2TF = lambda xbool : 'T' if xbool else 'F'
    with open(fscript,'w') as f:
        f.write('#!/bin/bash\n')
        if mode == 'babsma':
            f.write('babsma_lu << EOF\n')
        elif mode == 'syn':
            f.write('bsyn_lu << EOF\n')
        elif mode == 'eqwidt':
            f.write('eqwidt_lu << EOF\n')
        if mode in ['babsma','syn']:
            f.write(f"'LAMBDA_MIN:'  '{wvl_minmax[0]}'\n")
            f.write(f"'LAMBDA_MAX:'  '{wvl_minmax[1]}'\n")
            f.write(f"'LAMBDA_STEP:' '{dwvl_step}'\n")
        if mode == 'babsma':
            f.write(f"'MODELINPUT:'  '{marcs_mod_file}'\n")
            f.write(f"'MODELOPAC :'  '{fresult}'\n")
            f.write(f"'XIFIX:'      '{f2TF(True)}'\n")
            f.write(f"{vt}\n")
        else:
            f.write(f"'MODELOPAC :'  '{ts_opac_file}'\n")
            f.write(f"'RESULTFILE:'  '{fresult}'\n")
            f.write(f"'NFILES  :'    '{len(flinelists)}'\n")
            for flinelist in flinelists:
                f.write(flinelist+'\n')
            f.write(f"'SPHERICAL:'    '{f2TF(spherical)}'\n")
            f.write('  30\n  300.00\n  15\n  1.30\n')
            f.write(f"'INTENSITY/FLUX  :' {ts_input['INTENSITY_FLUX']}")
            if len(isotopes) > 0:
                f.write(f"'ISOTOPES: {len(isotopes)}'\n")
                for isospecies,abundance in isotopes.items():
                    f.write(f"{isospecies} {abundance}\n")
        f.wirte(f"'METALLICITY:'  '{feh}'\n")
        f.write(f"'HELIUM    :'  '{ts_input['HELIUM']}'\n")
        f.write(f"'ALPHA/FE  :'  '{ts_input['ALPHA_FE']}'\n")
        f.write(f"'HELIUM    :'  '{ts_input['HELIUM']}'\n")
        f.write(f"'S-PROCESS :'  '{ts_input['S_PROCESS']}'\n")
        f.write(f"'R-PROCESS :'  '{ts_input['R_PROCESS']}'\n")
        if len(abundances) > 0:
            f.write(f"'INDIVIDUAL ABUNDANCES: {len(abundances)}'\n")
            for atomnum,abundance in abundances.items():
                f.write(f"{atomnum} {abundance}\n")
        f.write('EOF\n\end')
    os.system('chmod +x '+fscript)
    os.system(fscript)       
