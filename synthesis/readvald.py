import pandas


class Linelist(pandas.DataFrame):
    '''
    Class to store a linelist. It currently supports VALD formats
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
def read_valdshort(filename):
    linelist = []
    with open(filename, 'r') as f:
        line = f.readline()
        extract_stellar = 'lines selected,' in line # Linelist extracted from stellar has ''lines selected,' in the header
        while (not 'Ion' in line):
            # Read until the header, which should contain 'Ion'
            line = f.readline()
        line = f.readline()
        while (line.count(',')>5):
            values = line.split(',')
            if extract_stellar:
                species = values[0]
                wavelength = float(values[1])
                expot = float(values[2])
                loggf = float(values[4])
                gamrad = float(values[5])
                gamqst = float(values[6])
                gamvw = float(values[7])
                lande = float(values[8])
                references = values[10]
            else:
                species = values[0]
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
        linelist = Linelist(linelist, \
            columns=['species', 'wavelength', 'expot', 'loggf',\
                'gamrad', 'gamqst', 'gamvw', 'lande', 'references','HFS'])
    return linelist

def read_valdlong(filename):
    linelist = []
    with open(filename, 'r') as f:
        line = f.readline()
        extract_stellar = 'lines selected,' in line # Linelist extracted from stellar has ''lines selected,' in the header
        while (not 'Ion' in line):
            # Read until the header, which should contain 'Ion'
            line = f.readline()
        line = f.readline()
        while (line.count(',')>5):
            values = line.split(',')
            if extract_stellar:
                species = values[0]
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
            else:
                species = values[0]
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
            linelist.append(\
                [species, wavelength, loggf, expot_lo, j_lo, expot_up, j_up, \
                 lande_lo, lande_up, lande, gamrad, gamqst, gamvw, \
                coupling_lo, electron_conf_lo,  term_desig_lo, \
                coupling_up, electron_conf_up, term_desig_up, references, hfs])
            line = f.readline()
        linelist = Linelist(linelist, \
            columns=['species', 'wavelength', 'loggf', 'expot', 'j_lo', 'expot_up', 'j_up', \
                 'lande_lo', 'lande_up', 'lande', 'gamrad', 'gamqst', 'gamvw', \
                'coupling_lo', 'electron_conf_lo', 'term_desig_lo',
                'coupling_up', 'electron_conf_up', 'term_desig_hi', 'references', 'HFS'])
    return linelist