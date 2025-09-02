import pandas
from solarabundance21 import getamass, elemtopnum,isotope,getisotope
import numpy as np
from astropy.constants import k_B,a0
import astropy.units as u
from scipy.special import gamma
import csv
import re

class Linelist(pandas.DataFrame):
    '''
    Class to store a linelist. It currently supports VALD formats
    '''
    def __init__(self, *args, scaled = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaled = scaled

    def get_scaling(self):
        '''
        Get corrections for log-gf values for isotopes
        '''
        self["isotope_correction"] = 0.0
        
        for ii in range(5):
            self[f"fraction{ii+1}"] = 1.0
            za_unique = np.unique(self[[f"Z{ii+1}",f"A{ii+1}"]].values, axis=0)
            for za in za_unique:
                zz, aa = za
                mask = (self[f"Z{ii+1}"]==zz)&(self[f"A{ii+1}"]==aa)
                if aa == 0:
                    continue
                iso_fraction = getisotope(zz,aa)
                self.loc[mask,"isotope_correction"] += np.log10(iso_fraction)
                self.loc[mask,f"fraction{ii+1}"] = iso_fraction
                
    def modify_scaling(self, zz, amass_frac_dict):
        assert "isotope_correction" in self.columns, \
            "isotope_correction column not found. Call get_scaling first"
        frac_sum = np.sum(list(amass_frac_dict.values()))
        print(frac_sum)
        for key in amass_frac_dict.keys():
            amass_frac_dict[key] /= frac_sum

        for ii in range(5):
            mask = self[f"Z{ii+1}"]==zz
            if np.sum(mask) == 0:
                continue
            aa_unique = np.unique(self.loc[mask,f"A{ii+1}"].values)
            for aa in aa_unique:
                if aa == 0:
                    continue
                mask2 = (self[f"Z{ii+1}"]==zz)&(self[f"A{ii+1}"]==aa)
                if aa in amass_frac_dict.keys():
                    new_frac = amass_frac_dict[aa]
                else:
                    new_frac = 1e-99
                    print(f"Atomic mass {aa} not found in amass_frac_dict. Assuming that it doesn't exist")
                frac_original = np.unique(self[f"fraction{ii+1}"].values[mask2])
                if len(frac_original) != 1:
                    print("Warning: more than one fraction found for the same isotope")
                delta_frac = new_frac / frac_original
                self.loc[mask2,"isotope_correction"] += np.log10(delta_frac)
                self.loc[mask2,f"fraction{ii+1}"] = new_frac

    def apply_scaling(self):
        if self.scaled and "loggf_unscaled" in self.columns:
            print("Linelist is already scaled but unscaled loggf is available.")
            print("Overwriting the loggf values")
            self.loc[:,"loggf"] = self["loggf_unscaled"] + self["isotope_correction"]
            self.scaled = True
            return
        elif self.scaled:
            raise ValueError("Linelist is already scaled and no unscaled loggf is available")
        elif "isotope_correction" not in self.columns:
            raise ValueError("Linelist is not scaled and no isotope_correction column found"+\
                             "Please call get_scaling first")
        else:
            self.loc[:,"loggf_unscaled"] = self.loc[:,"loggf"].values
            self.loc[:,"loggf"] = self["loggf_unscaled"] + self["isotope_correction"]
            self.scaled = True
            return       
      


def convert_sigma_alpha_to_gamma(elem_num, sigma, alpha,t0=1.0e4*u.K):
    elemnum = int(elem_num)
    reduced_mass = 1.0 / \
        ( 1.0/getamass(1,pnum='yes') + 1.0/getamass(elemnum,pnum='yes'))
    vbar = np.sqrt(8.0*k_B*t0 / (np.pi*reduced_mass*u.u))
    v0 = 1.0e4*u.m/u.s
    sigma = sigma*a0**2.
    ww = (4.0/np.pi)**(alpha/2.)*gamma((4.0-alpha)/2.0)*vbar*sigma*((vbar/v0).decompose())**(-alpha)
    return np.log10((2.*ww/(u.cm**3/u.s)).decompose())


def get_atom_num(sp1):
    """
    This function does somehting like the following
    Ba 2 -> ['Ba'] [56] 2
    C2 1 -> ["C","C"] [6,6] 1
    CH -> ["C","H"] [6,1] 1
    """
    atom_mol = sp1.split()[0]
    if len(sp1.split()) == 1:
        ion = 1 # Assume it is neutral
    else:
        ion = int(sp1.split()[1])

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


def get_iso_atom_num(sp1):
    """
    This function does somehting like the following
    (12)C(12)C' -> ['C','C'] [6,6] [12,12] [0]
    (49)TiO' -> ['Ti','O'] [22,8] [49,0] [0]
    (135)Ba+' -> ['Ba'] [56] [135] [1]
    """
    atoms = []
    isos = []
    iso_tmp = 0
    
    stats = -1 
    # -1 for doing nothing
    # 0 for reading atoms, 
    # 1 for reading isotopes, 
    # 2 for reading ioniation stage
    idx = 0
    def append_one(iso1):
        atoms.append(sp1[idx_start:idx])
        isos.append(iso1)

    while (idx<len(sp1)) and (sp1[idx]!="'"):
        if sp1[idx] == "(":
            if stats == 0:
                append_one(iso_tmp)
                iso_tmp = 0
            stats = 1
            idx_start = idx+1
        elif sp1[idx] == ")":
            assert stats == 1, "Found ), so it must be reading the atomic mass"
            stats = -1
            iso_tmp = int(sp1[idx_start:idx])
        elif sp1[idx].isupper():
            if stats == 0:
                append_one(iso_tmp)
                iso_tmp = 0
            stats = 0
            idx_start = idx
        elif sp1[idx] == "+":
            if stats == 0:
                append_one(iso_tmp)
                iso_tmp = 0
            stats = 2
            idx_start = idx
        # Otherwise it is reading the second characters of atomic symbols
        idx += 1
    if stats == 0:
        append_one(iso_tmp)
        iso_tmp = 0
        ion = 1
    elif stats == 2:
        ion = idx - idx_start + 1 # [1, 2, 3]  for ["", +, ++]
    else:
        raise IOError("An unexpected end state", sp1, stats)
    zz = [elemtopnum(atom) for atom in atoms]
    return atoms, zz, isos, ion


def construct_columns(line):
    possible_names = {\
        "species":("Ion",),
        "wavelength":("WL"),
        "expot":("E_low","Excit",),
        "expot_up":("E_up",),
        "j_lo":("J lo",),
        "j_up":("J up",),
        "lande_lo":("lower",),
        "lande_up":("upper",),
        "loggf":("log gf",),
        "gamma_rad":("Rad.",),
        "gamma_stark":("Stark",),
        "gamma_vw":("Waals",),
        "lande":("factor","mean"),
        "vturb":("Vmic",),
        "depth":("depth",),
        "references":("References","Reference"),       
        }
    columns_names = list(possible_names.keys())
    pos_columns = []
    for val  in possible_names.values():
        pos = -1
        for v in val:
            pos_thiscan = line.find(v)
            if pos_thiscan>=0:
                pos = pos_thiscan
                break
        pos_columns.append(pos)
    columns = []
    for idx in np.argsort(pos_columns):
        pos = pos_columns[idx]
        if pos < 0:
            continue
        else:
            columns.append(columns_names[idx])
    return columns

def read_vald(filename): # for backwards compatibility
    return readvald(filename)


def readvald(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    ipos = 0
    while not ("Ion" in lines[ipos]):
        ipos += 1
        if ipos == len(lines):
            raise ValueError("No header found")
    columns1 = construct_columns(lines[ipos])

    islong = not "references" in columns1

    # If stellar long format, there is somehow a comma at the end of each line
    if islong and ("depth" in columns1):
        columns1.append("empty")

    ncol = len(columns1)        
    data_loc = [i for i in range(len(lines)) if lines[i].count(',') >= ncol-1] # >= becasue there might be a comma in the refenreces column
    assert len(data_loc) > 0, "No data found"

    # first line
    data_lines = [lines[i] for i in data_loc]
    data = pandas.read_csv(csv.StringIO(''.join(data_lines)), header=None,quotechar="'",names=columns1,skipinitialspace=True)
    assert len(columns1) == len(data.columns), f"Number of columns in the header ({len(columns1)}) does not match the number of columns in the data ({len(data.columns)})"
    
    if islong:   
        # second row
        data.loc[:,"details_lo"] = [lines[i+1].strip() for i in data_loc]

        # Thrid row
        data.loc[:,"details_up"] = [lines[i+2].strip() for i in data_loc]

        # Fourth row
        data.loc[:,"references"] = [lines[i+3].strip() for i in data_loc]

    data.loc[:,"HFS"] = ["hfs" in ref for ref in data["references"].values]
    data.sort_values("wavelength",inplace=True)
    linelist = Linelist(data)

    # Need to check if isotopic ratios are already scaled or not
    linelist.scaled = \
        any(["oscillator strengths were scaled" in ll for ll in lines])
    if not linelist.scaled:
        if not any(["oscillator strengths were NOT scaled" in ll for ll in lines]):
            print(\
                "The linelist does not have the scaling information"+\
                "I assume that the linelist is not scaled but this might be wrong")
   

    linelist["Z1"] = np.zeros(len(linelist)).astype(int)
    linelist["Z2"] = np.zeros(len(linelist)).astype(int)
    linelist["Z3"] = np.zeros(len(linelist)).astype(int)
    linelist["Z4"] = np.zeros(len(linelist)).astype(int)
    linelist["Z5"] = np.zeros(len(linelist)).astype(int)
    linelist["A1"] = np.zeros(len(linelist)).astype(int)
    linelist["A2"] = np.zeros(len(linelist)).astype(int)
    linelist["A3"] = np.zeros(len(linelist)).astype(int)
    linelist["A4"] = np.zeros(len(linelist)).astype(int)
    linelist["A5"] = np.zeros(len(linelist)).astype(int)
    linelist["ion"] = np.zeros(len(linelist)).astype(int)

    pattern = re.compile(r"\(\d{1,3}\)[A-Z][a-z]?") # inspired by Korg's treatment of the Vald format

    isotope_species = []
    for idx in linelist.index:
        ref = linelist.loc[idx,"references"]
        match = pattern.search(ref)
        if match:
            istart = match.start()
            iend = ref.find(" ",istart)
            if iend == -1:
                iend = len(ref)
            isotope_species.append(ref[istart:iend])
        else:
            isotope_species.append("")
    isotope_species = np.array(isotope_species)

    for sp1 in np.unique(isotope_species):
        if sp1 == "":
            continue
        out1 = get_iso_atom_num(sp1)
        for ii,zzaa in enumerate(zip(out1[1],out1[2])):
            linelist.loc[isotope_species==sp1,f"Z{ii+1}"] = zzaa[0]
            linelist.loc[isotope_species==sp1,f"A{ii+1}"] = zzaa[1]
        linelist.loc[isotope_species==sp1,"ion"] = out1[3]
    
    mask_noiso = (linelist["Z1"]==0)
    for sp1 in np.unique(linelist["species"].values):
        out1 = get_atom_num(sp1)
        linelist.loc[(linelist["species"]==sp1) & mask_noiso,"ion"] = out1[2]    
        for ii,zz in enumerate(out1[1]):
            linelist.loc[(linelist["species"]==sp1)&mask_noiso,f"Z{ii+1}"] = zz
    return linelist

def read_valdshort(filename):# for backwards compatibility
    return read_vald(filename)

def read_valdlong(filename): # for backwards compatibility
    return read_vald(filename)
