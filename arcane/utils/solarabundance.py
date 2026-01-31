import numpy as np
import os
import glob

atoms = ['','H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'I', 'Te', 'Xe', 'Cs',
'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl',
'Mc', 'Lv', 'Ts', 'Og']

solar_abundance = np.zeros(len(atoms)) - 99

dir_path = os.path.join(\
    os.path.dirname(os.path.realpath(__file__)),
    "tables_solarabundance")
available_versions = glob.glob(\
    os.path.join(\
    dir_path,
    "*.csv"
    )
)
available_versions = [os.path.basename(a).replace(".csv","") for a in available_versions]
current_version = ""

def load_solarabundance(version="Asplund2021"):
    global solar_abundance
    global current_version
    solar_abundance[:] = -99 # Reset all abundances
    assert version in available_versions,\
        f"Specified solar abundance version, {version} not recognized."+\
        f" Available versions are: {available_versions} at {dir_path}"
    table_path = os.path.join(\
        dir_path,
        f"{version}.csv"
    )
    with open(table_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        i, a = l.split(',')[0:2]
        solar_abundance[int(i)] = float(a)
    current_version = version

load_solarabundance()
    

def get_atomnum(elem):
    if elem not in atoms:
        raise ValueError(f"Element {elem} not recognized.")
    return atoms.index(elem)

def get_elemname(atomnum):
    return atoms[atomnum]

def _get_solar(elem):
    try:
        int(elem)
    except:
        elem = get_atomnum(elem)
    return solar_abundance[elem]

get_solar = np.vectorize(_get_solar)
    
