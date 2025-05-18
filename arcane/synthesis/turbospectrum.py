import shutil
import os
import warnings
from arcane.mdlatm import marcs
import numpy as np
from arcane.synthesis.linelist import get_atom_num
from arcane.synthesis.readvald import convert_sigma_alpha_to_gamma,Linelist
from arcane.utils.solarabundance import get_atomnum
from arcane.utils import utils
from arcane.mdlatm import marcs,avg3d
import pandas

ts_default_input = {
    'INTENSITY_FLUX':'Flux',
    'R_PROCESS':0.0,
    'S_PROCESS':0.0,
    'ALPHA_Fe' :0.0,
    'HELIUM' : 0.0,
    'ABFIND' : True,
    'PURE-LTE': False,
}

ts_nocheck_input = ["ABUND_SOURCE","NLTE","NLTEINFOFILE","MODELATOMFILE",
    "DEPARTUREFILE","DEPARTBINARY","SEGMENTSFILE","RESOLUTION",
    "TSUJI","LOGFILE","ATOMDATA","MODELOPAC","PARAMETER","SPECDATA",
    "MOLECULES","LIMBDARK","MULTIDUMP","CONTINUOUS-OPACITIES",
    "C_WAVELENGTHS","DYDRODYN_DEPTH","COS(THETA)","SCATTFRAC",
    ]# These are recognized as input but not will be checked

def create_ts_species_id(atoms, isos=None):
    """
    Create a Turbospectrum species ID from atomic numbers and isotopes.
    
    Parameters
    ----------
    atoms : list of int
        List of atomic numbers.
    isos : list of int, optional
        List of isotopes. If None, defaults to the first isotope for each atom.
    
    Returns
    -------
    str
        The Turbospectrum species ID as a string.
    """
    atoms = [z for z in atoms if z > 0]
        
    if isos is None:
        isos = [0] * len(atoms)
    else:
        isos = [isos[i] for i in range(len(atoms)) if atoms[i] > 0]
    
    argsort = np.argsort(atoms)
    atoms = [atoms[i] for i in argsort]
    isos = [isos[i] for i in argsort]
    
    atom_str = ''.join([str(atom).zfill(2) for atom in atoms])
    iso_str = ''.join([str(iso).zfill(3) for iso in isos])
    
    return f"{atom_str}.{iso_str}"

def get_atom_num_isos_ts(species_id):
    """
    Get the atomic number and isotopes for a given Turbospectrum species ID.
    
    Parameters
    ----------
    species_id : str, float, int
        The MOOG species ID 
        (e.g., '106.0', '106.001013', '56.0', '56.137', "106", "56" etc.).
    
    Returns
    -------
    tuple
        A tuple containing the atomic number and a list of isotopes.
    """
    
    if type(species_id) in [int, float]:
        species_id = str(species_id)
        warnings.warn(f"species_id is a number, converting to string: {species_id}\n"+\
            "This might cause an issue. Make sure to check the result.")
        
    # Remove leading and trailing whitespace
    species_id = species_id.lstrip().rstrip()
    
    if "." in species_id:
        dot_loc = species_id.find(".")
        atommol_id = species_id[:dot_loc]
    else:
        dot_loc = len(species_id)
        atommol_id = species_id

    atom1 = -1
    end_loc = dot_loc
    atoms = []
    while (atom1 != 0) & (end_loc > 0):
        start_loc = end_loc - 2
        if start_loc < 0:
            start_loc = 0
        atom1 = int(atommol_id[start_loc:end_loc])
        if atom1 > 0:
            atoms.append(atom1)
        end_loc = start_loc
    atoms.reverse()
    
    if len(species_id) > dot_loc + 1:
        isos = [int(species_id[dot_loc+1+i*3:dot_loc+1+i*3+3]) \
            for i in range(len(atoms))]            
    else:
        isos = list(np.zeros(len(atoms),dtype=int))  
    return atoms, isos

def write_linelist(linelist, flinelist, ignore_isotope=True, defalut_dampnum=2.5):
    """
    Write the TS linelist file from the linelist, which can be a pandas dataframe,
    a dictionary, or a Linelist object. 
    
    Parameters
    ----------
    linelist : Linelist, pandas.DataFrame, dict
        The linelist to be written. It can be a Linelist object, a pandas DataFrame,
        or a dictionary.
        It needs to have wavelength, loggf, expot, and one of ("ts_species", "species, "Z1"..."Z5""ion")
        
    flinelist : str
        The name of the output file.
        The file will be created if it does not exist.
        If it exists, it will be overwritten.
    
    ignore_isotope : bool, optional    
        If ignore_isotope is True, all the species are treated as .000, 
        assuming that the linelist has already been scaled. Even if it is 
        set to False, if the linelist is already scaled (judged by the 
        linelist.scaled attribute), the isotopes are ignored.

    """
    # The linelist is not scaled, and we should not ignore the isotopes
    # gf-values will be scaled by the turbospectrum by its default 
    # isotope ratio, which I don't personally like, so make sure you scale
    # the linelist before passing it to this function
    flag_isotope = not ignore_isotope and hasattr(linelist,"scaled") and linelist.scaled
    
    if flag_isotope:
        raise ValueError("Writing the linelist with isotope information is not supported yet")
    
    if type(linelist) is pandas.DataFrame:
        linelist = linelist.to_dict(orient="list")
    
    for key in ["wavelength", "loggf", "expot"]:
        assert key in linelist.keys(), f"{key} needs to be in linelist keys"
    
    if "ts_species" not in linelist.keys():
        if not "Z1" in linelist.keys():
            assert "species" in linelist.keys(), \
                "You have to specify species by ts_species, Z1..Z5, or species"
            species_arr = np.array(linelist["species"])
            species_unique = np.unique(linelist["species"])
            linelist["Z1"] = np.zeros(len(linelist["species"]),dtype=int)
            linelist["Z2"] = np.zeros(len(linelist["species"]),dtype=int)
            linelist["Z3"] = np.zeros(len(linelist["species"]),dtype=int)
            linelist["Z4"] = np.zeros(len(linelist["species"]),dtype=int)
            linelist["Z5"] = np.zeros(len(linelist["species"]),dtype=int)
            linelist["ion"] = np.zeros(len(linelist["species"]),dtype=int)
            for sp in species_unique:
                mask = species_arr == sp
                atoms, zz, ion = get_atom_num(sp)
                linelist["Z1"][mask] = zz[0]
                linelist["Z2"][mask] = zz[1]
                linelist["Z3"][mask] = zz[2]
                linelist["Z4"][mask] = zz[3]
                linelist["Z5"][mask] = zz[4]
                linelist["ion"][mask] = ion
        # Now everythiing should have Z1..Z5 and ion
        assert "ion" in linelist.keys(), "ion needs to be in linelist keys"
        isotopes = [None] * len(linelist["Z1"])
        if flag_isotope:
            assert "A1" in linelist.keys(), "A1..A5 needs to be in linelist keys"
            isotopes = [list(aa) for aa in zip(
                linelist["A1"], linelist["A2"],
                linelist["A3"], linelist["A4"],
                linelist["A5"])]
        atoms = [list(zz) for zz in zip(
            linelist["Z1"], linelist["Z2"],
            linelist["Z3"], linelist["Z4"],
            linelist["Z5"])]
        linelist["ts_species"] = [
            create_ts_species_id(atom_seq, iso_seq)
            for atom_seq, iso_seq in zip(atoms, isotopes)
        ]
    if not flag_isotope:
        linelist["ts_species"] = [
            ts_sp.split(".")[0] + "." + ["0"] * len(ts_sp.split(".")[1])
            for ts_sp in linelist["ts_species"]
        ]
    ts_species_unique = np.unique(linelist["ts_species"]) 
    ts_species_arr = np.array(linelist["ts_species"])
    
    if "j_up" in linelist.keys():
        gup = 2 * linelist["j_up"] + 1
    else:
        gup = 1
    if "gamma_vw" in linelist.keys():
        fdamp = linelist["gamma_vw"]
        fdamp = np.where(fdamp==0.0, defalut_dampnum, fdamp)
    else:
        fdamp = np.ones(len(linelist["wavelength"])) * defalut_dampnum

    if not "gamma_rad" in linelist.keys():
        linelist["gamma_rad"] = np.zeros(len(linelist["wavelength"]))
    if not "gamma_stark" in linelist.keys():
        linelist["gamma_stark"] = np.zeros(len(linelist["wavelength"]))
    if not "ew" in linelist.keys():
        linelist["ew"] = np.zeros(len(linelist["wavelength"]))
    if not "ew_err" in linelist.keys():
        linelist["ew_err"] = np.zeros(len(linelist["wavelength"]))
        
    
    wmin, wmax = -1,-1
    with open(flinelist, 'w') as f:
        for ts_sp in ts_species_unique:
            mask = ts_species_arr == ts_sp
            ions = np.unique(np.array(linelist["ion"])[mask])
            for ion in ions:
                mask_ion = np.array(linelist["ion"]) == ion
                mask_comb = mask & mask_ion
                nline = np.sum(mask)
                f.write(f"'{ts_sp}'  {ion:d} {nline:d}\n")
                f.write("\n")# This seems to be just a comment
                wvl = np.array(linelist["wavelength"])[mask_comb]
                if wmin < 0 or wvl.min() < wmin:
                    wmin = wvl.min()
                if wmax < 0 or wvl.max() > wmax:
                    wmax = wvl.max()
                for i in np.argsort(wvl):
                    f.write(\
                        f"{wvl[mask_comb][i]:.3f}  "+\
                        f"{linelist['expot'][mask_comb][i]:.3f}  "+\
                        f"{linelist['loggf'][mask_comb][i]:.3f}  "+\
                        f"{gup[mask_comb][i]:.3f}  "+\
                        f"{10.**linelist['gamma_rad'][mask_comb][i]:.3e}  "+\
                        f"{linelist['gamma_stark'][mask_comb][i]:.3f}  "+\
                        f" 'x' 'x'"+\
                        f"{linelist['ew'][mask_comb][i]:.3f}  "+\
                        f"{linelist['ew_err'][mask_comb][i]:.3f}"                        
                        )
                    # I'm using x and x for now as i don't save this information
                    # when reading a vald linelist.
                    # I also don't support NLTE for now.
    return wmin, wmax
                
        
            

    
    
            

def run_turbospectrum(mode,
    linelist = None, run_id='',workdir='.',
    ts_opac_file = None, marcs_mod_file = None,
    teff = None, logg = None, feh = None, alphafe = None, 
    feh_mod = None, alphafe_mod = None,
    mdlatm_io = "marcs",
    vt = None,
    defalut_dampnum = 3.,
    wmin = None, wmax = None,
    dwvl_margin = 2.0,
    dwvl_step = 0.01,
    spherical = None,
    **kw_args):
    '''
    Run turbospectrum. 
    
    See TURBOSPECTRUM/source/input.f for input parameters.

    Parameters
    ----------
    mode : str
        'babsma', 'syn', 'eqwidt'
    '''
    mdlatm = globals()[mdlatm_io]
    
    string_in = f''

    if (mode != 'babsma'):
        if linelist is None:
            raise ValueError('linelist is needed for modes other than babsma')
        # Need a part that reads or writes the linelist and get the wavelength range
        wmin_ll, wmax_ll = write_linelist(linelist, flinelist, 
            defalut_dampnum = defalut_dampnum)
    
    

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
