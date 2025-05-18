import pandas
from solarabundance21 import getamass, elemtopnum,isotope,getisotope
import numpy as np
from astropy.constants import k_B,a0
import astropy.units as u
from scipy.special import gamma
import csv
import re
import warnings

# Everything in readvald should eventually be moved to this file for better organization.
# This is a temporary solution
# MOOG species id function should also utilize the functions currently in readvald 
from readvald import * 

def get_atom_num(species_id):
    """
    Get the atomic number for a given species ID.
    "Mg I" -> [12], 1
    "Mg II" -> [12], 2
    "Mg 1" -> [12], 1
    "Mg 2" -> [12], 2
    "Mg" -> [12], 1
    "MgH" -> [12, 1], 1
    "MgH 1" -> [12, 1], 1
    "C2" -> [6,6], 1
    "C2 1" -> [6,6], 1
    
    Parameters
    ----------
    species_id : str
        The species ID (e.g., "Mg I", "Mg II", "Mg 1", "Mg 2", "Mg", "MgH", "MgH 1", "C2", "C2 1").
    
    Returns
    -------
    atoms : list
        A list of atomic names.
    
    zz : list
        A list of atomic numbers corresponding to the atoms.
    
    ion : int
        The ionization state (1 for neutral, 2 for singly ionized, etc.).
    """
    species_id = species_id.lstrip().rstrip()
    
    if " " in species_id:
        # Split the species ID into atom/mol and ionization state parts
        atom_mol, ion = species_id.split(" ")
        if ion.isdigit():
            ion = int(ion)
        else:
            ion = ion.count("I")
    else:
        atom_mol = species_id
        ion = 1
    
    atoms = []        
    start_idx = -1
    for idx in range(len(atom_mol)):
        if atom_mol[idx].isupper():
            if start_idx >= 0:
                atoms.append(atom_mol[start_idx:idx])
            start_idx = idx
        elif atom_mol[idx].isnumeric():
            for ii in range(int(atom_mol[idx])):
                atoms.append(atom_mol[start_idx:idx])
            start_idx = -1
    if not atom_mol[idx].isnumeric():
        atoms.append(atom_mol[start_idx:])

    zz = [elemtopnum(atom) for atom in atoms]

    return atoms, zz, ion
    
    

def get_atom_num_isos_moog(species_id):
    """
    Get the atomic number and isotopes for a given MOOG species ID.
    
    Parameters
    ----------
    species_id : str, float, int
        The MOOG species ID (e.g., '106.0', '106.00113', '56.1', '56.11137', "106", "56" etc.).
    
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
        ion = int(species_id[dot_loc+1])
    else:
        dot_loc = len(species_id)
        atommol_id = species_id
        ion = 0

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
    
    if len(species_id) > dot_loc + 2:
        # If there are more than one digit after the dot, it has isotope information
        if len(atoms) == 1:
            isos = [int(species_id[dot_loc+2:dot_loc+5])]
        else:
            isos = [int(species_id[dot_loc+2+i*2:dot_loc+2+i*2+2]) \
                for i in range(len(atoms))]            
    else:
        isos = list(np.zeros(len(atoms),dtype=int))  
    return atoms, isos, ion +1 
    


def create_moog_species_id(atoms, ion, isos=None):
    """
    Create a MOOG species ID from atomic numbers, ionization state, and isotopes.
    
    Parameters
    ----------
    atoms : list of int
        List of atomic numbers.
    ion : int
        Ionization state (1 for neutral, 2 for singly ionized, etc.).
    isos : list of int, optional
        List of isotopes. If None, defaults to the first isotope for each atom.
    
    Returns
    -------
    str
        The MOOG species ID as a string.
    """
    
    if isos is None:
        isos = [0] * len(atoms)
        
    if len(atoms) != len(isos):
        raise ValueError("Length of atoms and isos must be the same.")
    
    argsort = np.argsort(atoms)
    atoms = [atoms[i] for i in argsort]
    isos = [isos[i] for i in argsort]
    
    if len(atoms) == 1:
        # Atoms
        atom_str = ''.join([str(atom).zfill(2) for atom in atoms])
        iso_str = ''.join([str(iso).zfill(3) for iso in isos])
    else:
        # molecules
        atom_str = ''.join([str(atom).zfill(2) for atom in atoms])
        iso_str = ''.join([str(iso).zfill(2) for iso in isos])
    
    return f"{atom_str}.{ion}{iso_str}"


