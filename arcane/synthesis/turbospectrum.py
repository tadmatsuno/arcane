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
import json

ts_default_input = {
    'INTENSITY/FLUX':'Flux',
    'R-PROCESS':0.0,
    'S-PROCESS':0.0,
    'ALPHA/Fe' :0.0,
    'HELIUM' : 0.0,
    'ABFIND' : True,
    'PURE-LTE': False,
    "LAMBDA_STEP": 0.01
}

ts_nocheck_input = ["ABUND_SOURCE","NLTE","NLTEINFOFILE","MODELATOMFILE",
    "DEPARTUREFILE","DEPARTBINARY","SEGMENTSFILE","RESOLUTION",
    "TSUJI","LOGFILE","ATOMDATA","MODELOPAC","PARAMETER","SPECDATA",
    "MOLECULES","LIMBDARK","MULTIDUMP","CONTINUOUS-OPACITIES",
    "C_WAVELENGTHS","DYDRODYN_DEPTH","COS(THETA)","SCATTFRAC",
    ]# These are recognized as input but not will be checked

ts_path = ""
home_path = os.path.expanduser("~")
src_path = os.path.join(home_path,".arcanesrc")

def find_turbospectrumfiles():
    global babsma_path, bsyn_path, DATA_path
    # This function reads the location of Turbospectrum files from the .arcanesrc file
    arcane_config = json.load(open(src_path,"r"))
    if not "turbospectrum_root" in arcane_config.keys():
        print("The root directory for Turbospectrum is not set in the .arcanesrc file")
        print("Call set_turbospectrum_path to set the path")
        return
    tsroot_path  = arcane_config["turbospectrum_root"]
    if not os.path.exists(tsroot_path):
        print("turbospectrum is not installed at:{0:s}".format(tsroot_path))
        print("Call set_turbospectrum_path to set the path")
        return

    # Searching for babsma
    for exec_dir in ["exec-ifx","exec-intel","exec-gf"]:
        babsma_path = os.path.join(tsroot_path,exec_dir,"babsma_lu")
        if os.path.exists(babsma_path):
            break
    if not os.path.exists(babsma_path):
        print("babsma_lu is not found in any of the exec* direcotries in "+\
              "the Turbospectrum directory: {0:s}".format(tsroot_path))
        print("Please check the installation of Turbospectrum")
        return
    
    # Searching for bsyn
    for exec_dir in ["exec-ifx","exec-intel","exec-gf"]:
        bsyn_path = os.path.join(tsroot_path,exec_dir,"bsyn_lu")
        if os.path.exists(bsyn_path):
            break
    if not os.path.exists(bsyn_path):
        print("bsyn_lu is not found in any of the exec* direcotries in "+\
              "the Turbospectrum directory: {0:s}".format(tsroot_path))
        print("Please check the installation of Turbospectrum")
        return
    
    # Searching for DATA
    DATA_path = os.path.join(tsroot_path,"DATA")
    if not os.path.exists(DATA_path):
        print("DATA directory is not found in the Turbospectrum directory: {0:s}".format(tsroot_path))
        print("Please check the installation of Turbospectrum")
        return    
    
    print("babsma_lu is found at: {0:s}".format(babsma_path))
    print("bsyn_lu is found at: {0:s}".format(bsyn_path))
    print("DATA directory is found at: {0:s}".format(DATA_path))
    print("Turbospectrum is ready to use")
    return

def set_turbospectrum_path(ts_path):
    if os.path.exists(ts_path):
        shutil.copy(src_path,src_path+"_old")
        arcane_setup = json.load(open(src_path,"r"))
        arcane_setup["turbospectrum_root"] = ts_path
        json.dump(arcane_setup,open(src_path,"w"))
    find_turbospectrumfiles()

find_turbospectrumfiles()


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

def read_linelist(flinelist):
    """
    Read Turbospectrum's linelist file and return a pandas.DataFrame    
    """
    
    with open(flinelist, "r") as f:
        lines = f.readlines()
        status = 0 # 0: header, >1=: data, -1: comment
        data = []
        for line in lines:
            if status == 0:
                ts_species, ion, nline = line.split()
                ion = int(ion)
                nline = int(nline)
                status = -1
            elif status > 0:
                fields = line.split()
                wvl = float(fields[0])
                expot = float(fields[1])
                loggf = float(fields[2])
                fdamp = float(fields[3])
                gup = float(fields[4])
                gamma_rad = float(fields[5])
                try:
                    gamma_stark = float(fields[6])
                    offset = 0
                except ValueError:
                    gamma_stark = 0.0
                    offset = -1                    
                level_up = fields[7+offset]
                level_lo = fields[8+offset]
                ew = float(fields[9+offset])
                ew_err = float(fields[10+offset])
                data.append({
                    "ts_species": ts_species,
                    "ion": ion,
                    "wavelength": wvl,
                    "expot": expot,
                    "loggf": loggf,
                    "fdamp": fdamp,
                    "gup": gup,
                    "gamma_rad": gamma_rad,
                    "gamma_stark": gamma_stark,
                    "level_up": level_up,
                    "level_lo": level_lo,
                    "ew": ew,
                    "ew_err": ew_err
                })
                status -= 1
            else:
                status = nline
        data = pandas.DataFrame(data)
    return data
        
    

def write_linelist(linelist, flinelist, ignore_isotope=True, default_dampnum=2.5,default_loggamrad=5):
    """
    Write the TS linelist file from the linelist, which can be a pandas dataframe,
    a dictionary, or a Linelist object. 
    
    Parameters
    ----------
    linelist : Linelist, pandas.DataFrame, dict
        The linelist to be written. It can be a Linelist object, a pandas DataFrame,
        or a dictionary.
        It needs to have wavelength, loggf, expot, and one of ("ts_species", "ion", "species, "Z1"..."Z5 & ion")
        
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
    
    
    if isinstance(linelist, pandas.DataFrame):
        linelist = linelist.to_dict(orient="list")
    
    for key in ["wavelength", "loggf", "expot"]:
        assert key in linelist.keys(), f"{key} needs to be in linelist keys"
    
    if "ts_species" in linelist.keys():
        # Just need to check if it has "ion" in the columns
        assert "ion" in linelist.keys(), "ionization status also needs to be in linelist keys"
    else:
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
            ts_sp.split(".")[0] + "." + "0" * len(ts_sp.split(".")[1])
            for ts_sp in linelist["ts_species"]
        ]
    
    if "j_up" in linelist.keys():
        gup = 2 * np.array(linelist["j_up"]) + 1
    else:
        gup = 1 * np.ones(len(linelist["wavelength"]))
    if "gamma_vw" in linelist.keys():
        #print("gamma_vw is in linelist")
        fdamp = np.array(linelist["gamma_vw"])
        fdamp = np.where(fdamp==0.0, default_dampnum, fdamp)
    else:
        fdamp = np.ones(len(linelist["wavelength"])) * default_dampnum

    if "gamma_rad" in linelist.keys():
        loggam_rad = np.array(linelist["gamma_rad"])
        loggam_rad = np.where(loggam_rad==0.0, default_loggamrad, loggam_rad)
    else:
        loggam_rad = np.ones(len(linelist["wavelength"])) * default_loggamrad
    if not "gamma_stark" in linelist.keys():
        linelist["gamma_stark"] = 0.0
    if not "ew" in linelist.keys():
        linelist["ew"] = np.zeros(len(linelist["wavelength"]))
    if not "ew_err" in linelist.keys():
        linelist["ew_err"] = np.zeros(len(linelist["wavelength"]))
#    print(linelist["ew"])
        
    
    wmin, wmax = -1,-1
    ts_species_unique = np.unique(linelist["ts_species"]) 
    ts_species_arr = np.array(linelist["ts_species"])
    with open(flinelist, 'w') as f:
        for ts_sp in ts_species_unique:
            mask = ts_species_arr == ts_sp
            ions = np.unique(np.array(linelist["ion"])[mask])
            for ion in ions:
                mask_ion = np.array(linelist["ion"]) == ion
                mask_comb = mask & mask_ion
                nline = np.sum(mask_comb)
                f.write(f"'{ts_sp}'  {ion:d} {nline:d}\n")
                f.write("comment\n")# It seems this can't be empty
                wvl = np.array(linelist["wavelength"])[mask_comb]
                if wmin < 0 or wvl.min() < wmin:
                    wmin = wvl.min()
                if wmax < 0 or wvl.max() > wmax:
                    wmax = wvl.max()
                use_stark = any(np.array(linelist["gamma_stark"])[mask_comb] > 0) and \
                    float(ts_sp) < 100 # Somehow stark format does not work for molecules
                for wvl1,expot1,loggf1,fdamp1,gup1,frad1,fstark1,ew1,ew_err1 in \
                    zip(np.array(linelist["wavelength"])[mask_comb],
                        np.array(linelist["expot"])[mask_comb],
                        np.array(linelist["loggf"])[mask_comb],
                        np.array(fdamp)[mask_comb],
                        gup[mask_comb],
                        np.array(loggam_rad)[mask_comb],
                        np.array(linelist["gamma_stark"])[mask_comb],
                        np.array(linelist["ew"])[mask_comb],
                        np.array(linelist["ew_err"])[mask_comb]):
                    f.write(f"{wvl1:.3f}  "+\
                        f"{expot1:.3f}  "+\
                        f"{loggf1:.3f}  "+\
                        f"{fdamp1:.3f}  "+\
                        f"{gup1:.3f}  "+\
                        f"{np.where(frad1>0,10.**frad1,frad1):.3e}  ")
                    if use_stark:
                        f.write(f"{fstark1:.3f}  ")
                    f.write(\
                        f" 'x' 'x' "+\
                        f"{ew1:.3f}  "+\
                        f"{ew_err1:.3f}\n"
                    )                        
                    # I'm using x and x for now as i don't save this information
                    # when reading a vald linelist.
                    # I also don't support NLTE for now.
    return wmin, wmax

def ts_input_format(value):
    if isinstance(value, bool):
        return "T" if value else "F"
    return str(value)

                
def run_turbospectrum(mode,
    linelist = None, run_id='',workdir='.',
    ts_opac_file = None, marcs_mod_file = None,
    teff = None, logg = None, feh = None, alphafe = None, 
    feh_mod = None, alphafe_mod = None,
    mdlatm_io = "marcs",
    vt = None,
    default_dampnum = 3.,
    wmin = None, wmax = None,
    spherical = None,
    **kw_args):
    '''
    Run turbospectrum. 
    
    See TURBOSPECTRUM/source/input.f for input parameters.

    Parameters
    ----------
    mode : str
        'babsma', 'syn'
    '''
    mdlatm = globals()[mdlatm_io]
    cwd = os.getcwd()
    if not os.path.exists(workdir):
        os.makedirs(workdir)
   
    
    if mode not in ['babsma','syn']:
        raise ValueError('mode should be babsma or syn')
   
    if isinstance(linelist,str):
        flinelist = linelist
    else:
        flinelist = f'{workdir}/linelist_{run_id}.lin'

    if marcs_mod_file is None:
        fmodelin = os.path.join(workdir, f"marcs_{run_id}.mod")
    else:
        fmodelin = marcs_mod_file

    if ts_opac_file is None:
        ts_opac_file = os.path.join(workdir, f"ts_{run_id}.opac")
    elif vt is not None:
        warnings.warn("vt is specified, but ts_opac_file is given. vt will be ignored.")

    fresult = os.path.join(cwd,workdir, f"ts_{run_id}.out")
    fparam = os.path.join(cwd,workdir, f"ts_{run_id}.param")
    
    # Make everything to abs path
    flinelist = os.path.join(cwd, flinelist)
    fmodelin = os.path.join(cwd,fmodelin)
    ts_opac_file = os.path.join(cwd,ts_opac_file)
    fresult = os.path.join(cwd,fresult)
    fparam = os.path.join(cwd,fparam)
    flog = os.path.join(cwd, workdir, f"ts_{run_id}.log")
    
    if (mode != 'babsma'):
        # checking linelist and running babsma will be done if it is not available
        if linelist is None:
            raise ValueError('linelist is needed for modes other than babsma')
        elif isinstance(linelist,str):
            if not os.path.exists(linelist):
                raise ValueError(f'linelist file {linelist} does not exist')
            # If wmin or wmax is not specified, we have to get them from the linelist
            if wmin is None or wmax is None:
                linelist = read_linelist(flinelist)
                if wmin is None:
                    wmin = linelist['wavelength'].min()
                if wmax is None:
                    wmax = linelist['wavelength'].max()
        else:
            wmin_ll, wmax_ll = write_linelist(linelist, flinelist, 
                default_dampnum = default_dampnum)
            if wmin is None:
                wmin = wmin_ll
            if wmax is None:
                wmax = wmax_ll
            
        if not os.path.exists(ts_opac_file):
            print("Calling babsma")
            run_turbospectrum("babsma", run_id=run_id+"OPAC", workdir=workdir,
                marcs_mod_file = marcs_mod_file, ts_opac_file = ts_opac_file,
                teff=teff, logg=logg, feh=feh, alphafe=alphafe,
                feh_mod=feh_mod, alphafe_mod=alphafe_mod,
                mdlatm_io=mdlatm_io, vt=vt,
                default_dampnum=default_dampnum,
                wmin = wmin-1, wmax = wmax+1,
                spherical=spherical,
                **kw_args)

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
            model = marcs.get_marcs_mod(teff, logg, feh_mod, \
                alphafe=alphafe_mod, outofgrid_error=True)
            marcs.write_marcs(fmodelin,model)
        if feh is None:
            feh = model['m_h']
        if alphafe is None:
            alphafe = model['alpha_m']
    else:
        if spherical is None:
            assert os.path.exists(fmodelin), \
                'marcs_mod_file specified does not exist'
            model = marcs.read_marcs(fmodelin)
            spherical = model["radius"] > 1
                    
        if (feh is None or alphafe is None) and (not marcs_mod_file is None):
            assert os.path.exists(fmodelin),'marcs_mod_file specified does not exist'
            model = marcs.read_marcs(fmodelin)
            if feh_mod is None:
                feh_mod = model['m_h']
            if alphafe_mod is None:
                alphafe_mod = model['alpha_m']
            if feh is None:
                feh = feh_mod
            if alphafe is None:
                alphafe = alphafe_mod

    # here create the ts_input dictionary
    abundances = {}
    isotopes = {}
    if "AX_dict" in kw_args.keys():
        for atomnum in kw_args["AX_dict"].keys():
            abundances[atomnum] = kw_args["AX_dict"][atomnum]
    if "Isotope_dict" in kw_args.keys():
        for isospecies in kw_args["Isotope_dict"].keys():
            assert isospecies is str, 'Isotope_dict keys should be str'
            isotopes[isospecies] = kw_args["Isotope_dict"][isospecies]
    ts_input = ts_default_input.copy()
    ts_input["LAMBDA_MIN"] = wmin
    ts_input["LAMBDA_MAX"] = wmax
    if "LAMBDA_STEP" in kw_args.keys():
        ts_input["LAMBDA_STEP"] = float(kw_args["LAMBDA_STEP"])
    for key in ts_default_input.keys():
        if key in kw_args.keys():
            ts_input[key] = kw_args[key]
    for key in ts_nocheck_input:
        if key in kw_args.keys():
            ts_input[key] = kw_args[key]
    for key in kw_args.keys():
        if key not in list(ts_default_input.keys()) + \
            list(ts_nocheck_input):
            warnings.warn(f"Input {key} is not recognized by Turbospectrum. It will be ignored.")

    assert feh is not None,'feh is not specified'
    ts_input["METALLICITY"] = feh
    if alphafe is not None:
        ts_input["ALPHA/Fe"] = alphafe
              
    for key, val in kw_args.items():
        if key.startswith("A_"):
            atomnum = int(key[2:])
            if atomnum in abundances.keys():
                warnings.warn(f"Abundance for atom {atomnum} is already set to {abundances[atomnum]}\n"+\
                    "A_X entries are prioritized over AX_dict inputs")
            abundances[atomnum] = val
            continue
        if key.startswith("I_"):
            isospecies = ".".join(key[2:].split("_"))
            if isospecies in isotopes.keys():
                warnings.warn(f"Isotope for species {isospecies} is already set to {isotopes[isospecies]}\n"+\
                    "I_X entries are prioritized over Isotope_dict inputs")
                isotopes[isospecies] = val
            continue


    
    command_ts = ""
    if mode == 'babsma':
        command_ts += babsma_path
    elif mode == 'syn':
        command_ts += bsyn_path
    command_ts += " > " + flog + " 2>&1 << EOF\n"
    
    # all the inputs in ts_input dictionary
    for key in ts_input.keys():
        command_ts += f"'{key}:' '{ts_input_format(ts_input[key])}'\n"
    if len(abundances) > 0:
        command_ts += f"'INDIVIDUAL ABUNDANCES: {ts_input_format(len(abundances))}'\n"
        for atomnum, abundance in abundances.items():
            command_ts += f"{atomnum} {abundance}\n"
    if len(isotopes) > 0:
        command_ts += f"'ISOTOPES: {ts_input_format(len(isotopes))}'\n"
        for isospecies, abundance in isotopes.items():
            command_ts += f"{isospecies} {abundance}\n"
            
        
    # babsma inputs
    if mode == "babsma":
        command_ts += f"'MODELINPUT:' '{fmodelin}'\n"+\
            f"'MODELOPAC:' '{ts_opac_file}'\n"+\
            f"'XIFIX: '{ts_input_format(True)}'\n"+\
            f"{vt}\n"
    # bsyn inputs
    if mode == "syn":
        command_ts += f"'MODELOPAC:' '{ts_opac_file}'\n"+\
            f"'RESULTFILE:' '{fresult}'\n"
        command_ts += f"'NFILES  :'    '{1}'\n"
        # currently only supporting one linelist
        command_ts += flinelist+'\n'
#        for flinelist in flinelists:
#            f.write(flinelist+'\n')
        command_ts += f"'SPHERICAL:'    '{ts_input_format(spherical)}'\n"
        command_ts += '  30\n  300.00\n  15\n  1.30\n'
    command_ts += "EOF" 

    with open(os.path.join(cwd,workdir,f"{run_id}.ts_input"), 'w') as f:
        f.write(command_ts)
    os.chdir(workdir)
    if not os.path.exists("DATA"):
        os.symlink( DATA_path, "DATA")
    os.system(command_ts)
    os.chdir(cwd)
