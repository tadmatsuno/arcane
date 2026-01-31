from . import solarabundance
import os
import glob
import numpy as np

dir_path = os.path.join(\
    os.path.dirname(os.path.realpath(__file__)),
    "tables_isotope")



available_sources = glob.glob(\
    os.path.join(
    dir_path,
    "*.csv"
    )
)
available_sources = [os.path.basename(a).replace(".csv","") for a in available_sources]

current_source = []
isotopic_ratio = np.zeros((len(solarabundance.atoms),296))

def load_isotopic_ratio(source="solar_Asplund2021"):
    """
    """
    global current_source
    global isotopic_ratio
    
    assert source in available_sources,\
        f"Specified isotopic ratio source, {source} not recognized."+\
        f" Available sources are: {available_sources} at {dir_path}"
    isotopic_ratio[:,:] = 0
    table_path = os.path.join(
        dir_path,
        f"{source}.csv"
    )
    with open(table_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        items = l.split(',')
        znum = int(items[0])
        amass = int(items[1])
        frac = float(items[2])
        isotopic_ratio[znum,amass] = frac

load_isotopic_ratio()

def _get_fraction(elem, amass):
    try:
        int(elem)
    except:
        elem = solarabundance.get_atomnum(elem)
    return isotopic_ratio[elem, amass]/100

def get_fraction(elem, amass):
    """
    Parameters
    ----------
    elem : str or int
        Element symbol or atomic number
        can be an array
    amass : int
        Atomic mass number
        can be an array
    """
    # First reshape elem and amass to be arrays of the same shape
    elem_arr = np.atleast_1d(elem)
    amass_arr = np.atleast_1d(amass)
    if elem_arr.shape != amass_arr.shape:
        if elem_arr.size == 1:
            elem_arr = np.full(amass_arr.shape, elem_arr.item())
        elif amass_arr.size == 1:
            amass_arr = np.full(elem_arr.shape, amass_arr.item())
        else:
            raise ValueError("elem and amass must have the same shape or be scalars")
    output = np.vectorize(_get_fraction)(elem_arr, amass_arr)
    try:
        return output.item()
    except:
        return output

    
isotope_mass = np.zeros((len(solarabundance.atoms),296))
atom_mass = np.zeros(len(solarabundance.atoms))
def load_atomic_mass(source="NIST"):
    """
    """
    global isotope_mass
    global atom_mass
    
    isotope_mass[:,:] = 0
    table_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "tables_atomicmass",
        f"{source}.csv"
    )
    with open(table_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        items = l.split(',')
        _znum = items[0]
        try:
            znum = int(_znum)
        except:
            pass
        amass = int(items[1])
        amu = float(items[2])
        isotope_mass[znum,amass] = amu
    atom_mass = np.sum(isotopic_ratio/100 * isotope_mass, axis=1)
load_atomic_mass()

def get_atomic_mass(elem):
    """
    Parameters
    ----------
    elem : str or int
        Element symbol or atomic number
    """
    try:
        int(elem)
    except:
        elem = solarabundance.get_atomnum(elem)
    return atom_mass[elem]
