from arcane_dev.spectrum import model
from arcane_dev.synthesis import moog,readvald
from arcane_dev.mdlatm import marcs,avg3d
from arcane_dev.utils import utils
import pandas
import sys
import os
from astropy.constants import c
import warnings
import iofiles
import numpy as np
import matplotlib.pyplot as plt
ckm = c.to('km/s').value

def _read_and_remove_comment(f):
    line = ''
    while len(line) ==0:
        line = f.readline().strip()
        if '#' in line:
            line = line[:line.find('#')].strip()
    return line
# parameters in batch_input:
# synth: the synthesis code to use
# linelist: linelist to use
# strong: linelist for strong lines to use
# workdir: working directory
# model_atm: model atmosphere type
# stars: a pandas dataframe containing the stellar parameters, teff, logg, vt, feh, (afe, A_*)
# action: the default action to perform. skip, fit, synth
# line_cont_parameters : a dictionary of default parameters for ContinuumAbsorptionModel, niterate and fit_vshift 
# line_parameters: a dictionary of default parameters for LineAbsorptionMode
# cont_parameters: a dictionary of default parameters for ContinuumAbsorptionModel
# continuum_region: Default continuum region is at n * cwvl / resolution away from the end of the line fitting region
# lines: a list of dictionaries containing the line information
def read_batch_input(filename):
    batch_input = {'line_cont_parameters':{},'line_parameters':{},'cont_parameters':{}}
    batch_input['line_parameters']['kw_fit_control'] = {}
    batch_input['lines'] = []

    line = ''
    with open(filename) as f:
        line = _read_and_remove_comment(f)
        while(line != 'END'):
            key = line.split(':')[0].strip()
            if key == 'CODE':
                value = line.split(':')[1].strip()
                batch_input['synth'] = globals()[value].synth
            elif key == 'N_LINELISTS':
                value = int(line.split(':')[1].strip())
                batch_input['linelist'] = []
                for i in range(value):
                    line = _read_and_remove_comment(f)
                    batch_input['linelist'].append(\
                        readvald.read_vald(line))
                batch_input['linelist'] = pandas.concat(\
                    batch_input['linelist'])
            elif key == 'WORKING_DIR':
                batch_input['workdir'] = line.split(':')[1].strip()
                if not os.path.exists(batch_input['workdir']):
                    os.makedirs(batch_input['workdir'])
            elif key == 'MODEL_ATM':
                batch_input['model_atm'] = line.split(':')[1].strip()
            elif key == 'STARS':
                value = line.split(':')[1].strip()
                batch_input['stars'] = pandas.read_csv(value,index_col='name')
                to_be_analyzed = []
                line = _read_and_remove_comment(f)
                while (line!='STAREND'):
                    key = line.split(':')[0].strip()
                    if key == 'STAR':
                        value = line.split(':')[1].strip()
                        star_name = value
                        if star_name == 'all':
                            to_be_analyzed = batch_input['stars'].index
                        else:
                            to_be_analyzed.append(star_name)
                            if star_name not in batch_input['stars'].index:
                                batch_input['stars'].loc[star_name] = {}
                    else:
                        value = float(line.split(':')[1].strip())
                        batch_input['stars'].loc[star_name,key] = value
                    line = _read_and_remove_comment(f)
            elif key =='DEFAULT_ACTION':
                batch_input['action'] = line.split(':')[1].strip()
            elif key == 'NITERATE_LINE_CONT':
                batch_input['line_cont_parameters']['niterate'] = int(line.split(':')[1].strip())
            
            elif key == 'FIT_POSITION':
                batch_input['line_cont_parameters']['fit_vshift'] = int(line.split(':')[1].strip())>0
            elif key == 'RESOLUTION':
                batch_input['line_parameters']['vfwhm_in'] = ckm/float(line.split(':')[1].strip())
            elif key == 'FIT_WIDTH':
                batch_input['line_parameters']['kw_fit_control']['fix_vfwhm'] = \
                    int(line.split(':')[1].strip()) <= 0
            elif key == 'NITERATE_LINE':
                batch_input['line_parameters']['niterate'] = int(line.split(':')[1].strip())
            elif key == 'LOW_REJ_LINE':
                batch_input['line_parameters']['low_rej'] = float(line.split(':')[1].strip())
            elif key == 'HIGH_REJ_LINE':
                batch_input['line_parameters']['high_rej'] = float(line.split(':')[1].strip())
            elif key == 'GROW_LINE':
                batch_input['line_parameters']['grow'] = float(line.split(':')[1].strip())
            elif key == 'GRID_SIZE':
                batch_input['line_parameters']['grid_size'] = float(line.split(':')[1].strip())
            elif key == 'SYN_MARGINE':
                batch_input['line_parameters']['syn_margine'] = [\
                    float(num1) for num1 in line.split(':')[1].strip().split(',')]
            elif key == 'ABUN_TOLERANCE':
                batch_input['line_parameters']['kw_fit_control']['xatol'] = float(line.split(':')[1].strip())
            elif key == 'WIDTH_LIMIT':
                batch_input['line_parameters']['kw_fit_control']['bounds_vfwhm'] = [\
                    float(num1) for num1 in line.split(':')[1].strip().split(',')]

            elif key == 'CONTINUUM':
                if line.split(':')[1].strip() == '1':
                    batch_input['cont_parameters']['order'] = -1
                elif line.split(':')[1].strip() == 'constant':
                    batch_input['cont_parameters']['order'] = 0
                elif line.split(':')[1].strip() == 'linear':
                    batch_input['cont_parameters']['order'] = 1
            elif key == 'CONTINUUM_REGION':
                batch_input['cont_parameters']['continuum_region'] = float(line.split(':')[1].strip())
            elif key == 'NITERATE_CONT':
                batch_input['cont_parameters']['niterate'] = int(line.split(':')[1].strip())
            elif key == 'LOW_REJ_CONT':
                batch_input['cont_parameters']['low_rej'] = float(line.split(':')[1].strip())
            elif key == 'HIGH_REJ_CONT':
                batch_input['cont_parameters']['high_rej'] = float(line.split(':')[1].strip())
            elif key == 'LINE':
                linename,act,wvl_region,fit_param,cont_region,ew = line.split(':')[1].split()[0:6]
                additional_info = {}
                if len(line.split(':')[1].split()) > 6:
                    for add_info in line.split(':')[1].split()[6:]:
                        key,value = add_info.split('=')
                        additional_info[key] = value
                line = _read_and_remove_comment(f)
                wvl_info_labels = ['moog_species','wvl','loggf','expot']
                delete = []
                add = []
                strong = []
                ew = []
                while (line!='0'):
                    info1 = {}
                    for ii,info in enumerate(line.split()[1:]):
                        if ii > 4:
                            key, value = info.split('=')
                            info1[key] = value
                        elif wvl_info_labels[ii] != 'moog_species':
                            info1[wvl_info_labels[ii]] = float(info)
                        else:
                            info1[wvl_info_labels[ii]] = info
                        
                    if line.split()[0] == 'd':
                        delete.append(info1.copy())
                    elif line.split()[0] == 'a':
                        add.append(info1.copy())
                    elif line.split()[0] == 's':
                        strong.append(info1.copy())
                    elif line.split()[0] == 'e':
                        ew.append(info1.copy())
                    line = _read_and_remove_comment(f)
                
                batch_input['lines'].append(\
                    {'linename':linename,'act':act,'wvl_region':wvl_region,'fit_param':fit_param,
                    'cont_region':cont_region,'ew':ew,**additional_info,
                    'delete':delete,'add':add,'strong':strong})                 

            else:
                print(f'Unknown key {key}')
            line = _read_and_remove_comment(f)
        return batch_input

def run1line(star,lineinfo,linecont_param,line_param,cont_param,synth_func,
             linelist,model_atm,workdir,default_action):
    
    line_region = lineinfo['wvl_region']
    line_region = line_region.replace('-',' ').replace(',','\n')
    line_region = utils.textsamples(line_region,reverse=True)
    wc = np.average(np.array([np.mean(ss) for ss in line_region]),
        weights=np.array([ss[1]-ss[0] for ss in line_region]))
    wmin = np.min([ss[0] for ss in line_region])
    wmax = np.max([ss[1] for ss in line_region])

    cont_region = lineinfo['cont_region']
    cont_region = cont_region.replace('-',' ').replace(',','\n')
    cont_region = utils.textsamples(cont_region,reverse=True)
    if np.all(np.array(cont_region)==0.0):
        dwvl = cont_param['continuum_region'] * wc * line_param['vfwhm_in']/ckm
        cont_region = [[wmin- dwvl, wmax + dwvl]]
    
    linelist_in = linelist[linelist['wavelength'] > (wmin - 2.*line_param['syn_margine'][0]) & \
                    (linelist['wavelength'] < (wmax + 2.*line_param['syn_margine'][1]) )]
    for line in lineinfo['delete']:# delete lines
        condition = np.ones(len(linelist_in),dtype=bool)
        if float(line['moog_species'])>-90:
            condition = condition & \
                (np.abs(linelist_in['moog_species'].astype(float) - float(linelist_in['moog_species']))<0.09)
        if line['wvl'] > -90:
            condition = condition & \
                (np.abs(linelist_in['wavelength'] - line['wvl'])<0.001)
        if line['loggf'] > -90:
            condition = condition & \
                (np.abs(linelist_in['loggf'] - line['loggf'])<0.001)
        if line['expot'] > -90:
            condition = condition & \
                (np.abs(linelist_in['expot'] - line['expot'])<0.001)
        if np.sum(condition) > 1:
            warnings.warn(f'Multiple lines found for the line to be deleted {line}')
        elif np.sum(condition) == 0:
            warnings.warn(f'No line found for the line to be deleted {line}')
        else:
            linelist_in = linelist_in[~condition]
    for line in lineinfo['add']:# add lines
        linelist_in = linelist_in.append(\
            {'wavelength':line['wvl'],
             'loggf':line['loggf'],
             'expot':line['expot'],
             'moog_species':line['moog_species']},
            ignore_index=True)
    if len(lineinfo['strong']) > 0:
        for line in lineinfo['strong']:# add strong lines
            condition = np.ones(len(linelist),dtype=bool)
            if float(line['moog_species'])>-90:
                condition = condition & \
                    (np.abs(linelist['moog_species'].astype(float) - float(line['moog_species']))<0.09)
            if line['wvl'] > -90:
                condition = condition & \
                    (np.abs(linelist['wavelength'] - line['wvl'])<0.001)
            if line['loggf'] > -90:
                condition = condition & \
                    (np.abs(linelist['loggf'] - line['loggf'])<0.001)
            if line['expot'] > -90:
                condition = condition & \
                    (np.abs(linelist['expot'] - line['expot'])<0.001)
            if np.sum(condition) > 1:
                warnings.warn(f'Multiple lines found for the line to be added as a strong line {line}')
            elif np.sum(condition) == 0:
                warnings.warn(f'No line found for the line to be added as a strong line {line}')
            else:
                strong = linelist[condition]
    else:
        strong = None


    syn_input_dict = {}
    for key in star.keys():
        if key == 'spec':
            # spec io here
            spec = iofiles.readspip(star[key])
        elif np.isfinite(star[key]):
            syn_input_dict[key] = star[key]
        elif key in ['teff','logg','vt','feh']:
            print('Missing stellar parameter')
            return
    syn_input_dict['mdlatm_io'] = model_atm
    syn_input_dict['run_id'] = f'{star.name}_{lineinfo["linename"]}'
    syn_input_dict['workdir'] = workdir
    syn_input_dict['linelist'] = linelist_in
    syn_input_dict['strong_lines'] = strong
    
    continuum = model.ContinuumPolynomial(\
        cont_param['order'],
        niterate=cont_param['niterate'],
        fit_mode='ratio',
        samples=cont_region)
    absorption = model.LineSynth1param(\
        synth_func,
        syn_input_dict,
        lineinfo['fit_param'],
        vfwhm=line_param['vfwhm_in'],
        grid_size=line_param['grid_size'],
        grid_scale='linear',
        niterate=line_param['niterate'],
        low_rej=line_param['low_rej'],
        high_rej=line_param['high_rej'],
        grow=line_param['grow'],
        naverage=1,
        fit_mode='subtract',
        samples=line_region,
        std_from_central=False,
        kw_fit_control=line_param['kw_fit_control'],
        )

    absorption.construct_grid()

    model_profile = model.ContinuumAbsorptionModel(
        model_absorption=absorption,
        model_continuum=continuum,
        niterate=linecont_param['niterate'],
        fit_vshift=linecont_param['fit_vshift']
    )
    model_profile.fit(spec[0],spec[1])
    return model_profile

def plot_result(f,model_profile,err=0.3):
    fitting_parameter = model_profile.model_absorption.update_synth_parameter

    fig, axs = plt.subplots(2,1,figsize=(10,5))
    ax = axs[0]
    wvl = model_profile.wavelength/(1.0+model_profile.vshift/ckm)
    ax.plot(wvl,
           model_profile.flux,
           'ko',ms=1.) # observation 
    ax.plot(wvl,
            model_profile.model_continuum(wvl),
            'C7--',lw=0.5,zorder=0.9) # Continuum
    ax.plot(wvl,model_profile.evaluate(wvl),'C0-',lw=1.0,zorder=1.0)
    used_absorption = model_profile.model_absorption.use_flag
    used_continuum = model_profile.model_continuum.use_flag
    samples_absorption = model_profile.model_absorption.samples
    samples_continuum = model_profile.model_continuum.samples
    wmin = np.minimum(np.min(np.array(samples_continuum)),
                        np.min(np.array(samples_absorption)))
    wmax = np.maximum(np.max(np.array(samples_continuum)),
                        np.max(np.array(samples_absorption)))
    delta_w = wmax - wmin
    wmin -= 0.1*np.maximum(delta_w,0.5)
    wmax += 0.1*np.maximum(delta_w,0.5)

    fitregion_absorption = utils.get_region_mask(wvl,samples_absorption)
    fitregion_continuum = utils.get_region_mask(wvl,samples_continuum)
    ax.plot(wvl[fitregion_absorption],model_profile.evaluate(wvl)[fitregion_absorption],\
            'r-',lw=1.0,zorder=1.0)
    ax.plot(wvl[fitregion_absorption&(~used_absorption)],
            model_profile.evaluate(wvl[fitregion_absorption&(~used_absorption)]),
            'rx',ms=2.0,zorder=1.0,lw=0.5)
    ax.plot(wvl[fitregion_continuum],model_profile.evaluate(wvl)[fitregion_continuum],\
            'b-',lw=1.0,zorder=1.0)
    ax.plot(wvl[fitregion_continuum&(~used_continuum)],
            model_profile.evaluate(wvl[fitregion_continuum&(~used_continuum)]),
            'bx',ms=2.0,zorder=1.0,lw=0.5)
    ymin = np.min([np.min(model_profile.flux[fitregion_absorption|fitregion_continuum]),
                np.min(model_profile.evaluate(wvl)[fitregion_absorption|fitregion_continuum])])
    ymax = np.max([np.max(model_profile.flux[fitregion_absorption|fitregion_continuum]),
                np.max(model_profile.evaluate(wvl)[fitregion_absorption|fitregion_continuum])])
    delta_y = ymax - ymin
    ymin -= 0.1*delta_y
    ymax += 0.1*delta_y
    best_fit = model_profile.model_absorption.synth_parameters[fitting_parameter]
    if err > 0:
        model_profile.model_absorption.synth_parameters[fitting_parameter] += err
        yerr1 = model_profile.evaluate(wvl)
        model_profile.model_absorption.synth_parameters[fitting_parameter] -= 2.*err
        yerr2 = model_profile.evaluate(wvl)
        ax.fill_between(wvl,yerr1,yerr2,color='C0',alpha=0.3,zorder=1)
    model_profile.model_absorption.synth_parameters[fitting_parameter] -= 50
    ax.plot(wvl,model_profile.evaluate(wvl,force_recompute=True),'C7--',lw=1.,zorder=1.)

    # Samples    
    ax.set(xlim=(wmin,wmax),ylim=(ymin,ymax))

    fig.savefig(f,dpi=300)       

def run_syntheses(fbatch_input):
    batch_input = read_batch_input(fbatch_input)
    for star_name in batch_input['stars'].index:
        star = batch_input['stars'].loc[star_name]
        for lineinfo in batch_input['lines']:
            if star_name in batch_input['stars'].index:
                if lineinfo['act'] == '0':
                    action = batch_input['action']
                if action == 'skip':
                    continue
                elif lineinfo['act'] == 'fit':
                    model_profile = run1line(star,lineinfo,
                        batch_input['line_cont_parameters'],
                        batch_input['line_parameters'],
                        batch_input['cont_parameters'],
                        batch_input['synth'],
                        batch_input['linelist'],
                        batch_input['model_atm'],
                        batch_input['workdir'],
                        batch_input['action'])
                    plot_result(os.path.join(batch_input['workdir'],\
                        star.name + '_' + lineinfo['linename'] + '.png'),                     
                        model_profile,err=0.3)
                else:
                    print(f'Unknown action {lineinfo["act"]} [Not implemented yet]')
            else:
                print(f'Star {star_name} not found in the stellar parameter list')
    return

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Usage: python synthesis_batch.py <batch_input_file>')
        sys.exit(1) 
    run_syntheses(sys.argv[1])