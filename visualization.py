#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.interpolate import interp1d



def get_peak_parameters_of_ITD_curve(x,y):
    _ITD = x
    _FR = y
    # interp
    _x  = _ITD
    _y = _FR
    _f = interp1d(_x, _y)
    _dt = 0.1
    _ITD_interp = np.arange(_x.min(),_x.max()+_dt,_dt)
    _ITD_interp = np.clip(_ITD_interp,_x.min(),_x.max())
    _FR_interp = _f(_ITD_interp)
    _smoothing_w_length = int(len(_FR_interp)/25)+1
    _smoothed_FR = savgol_filter(_FR_interp,_smoothing_w_length, 2)
    # find peaks 
    _peaks, _ = find_peaks(_smoothed_FR, prominence=1, width=20)
    _peak_heights = _smoothed_FR[_peaks]
    _peak = _peaks[_peak_heights==_peak_heights.max()]
    _ref_height = _smoothed_FR[_peak] * 0.5
    _half_width_seg = np.arange(len(_smoothed_FR))[_smoothed_FR>_ref_height][[0,-1]]
    _peak_ITD = _ITD_interp[_peak][0]
    _peak_FR = _smoothed_FR[_peak][0]
    _half_width = _ITD_interp[_half_width_seg][1] -_ITD_interp[_half_width_seg][0]
    return _peak_ITD, _peak_FR,_half_width

def interp_and_smooth_ITD_curve(x,y,scale = 25):
    _ITD = x
    _FR = y
    # interp
    _x  = _ITD
    _y = _FR
    _f = interp1d(_x, _y)
    _dt = 0.1
    _ITD_interp = np.arange(_x.min(),_x.max()+_dt,_dt)
    _ITD_interp = np.clip(_ITD_interp,_x.min(),_x.max())
    _FR_interp = _f(_ITD_interp)
    _smoothing_w_length = int(len(_FR_interp)/scale)+1
    _smoothed_FR = savgol_filter(_FR_interp,_smoothing_w_length, 2)
    return _ITD_interp,_smoothed_FR
    

def run_data_visualization():
    print('>> Compute peak parameters and plot figures')
    
    # load config
    _config_filename = 'config.pckl'
    with open(_config_filename, 'rb') as _file:
        _config = pickle.load(_file)
    import_path = _config['basic']['result_path']
    export_path = _config['basic']['fig_path']
    os.makedirs(export_path,exist_ok=True)
    os.makedirs(import_path,exist_ok=True)
    _dir_pckl = os.listdir(import_path)
    _sort_str = '.pckl'
    _dir_pckl = [x for x in _dir_pckl if x[-len(_sort_str):]==_sort_str]

    # list files
    df_files = pd.DataFrame(_dir_pckl,columns=['filename'])
    df_files['cate_name'] = df_files['filename'].apply(lambda x:x.split('.')[0])
    df_files['seed'] = df_files['cate_name'].apply(lambda x:int(x.split('_')[-1]))


    # load ITD dataset
    print('> loading ITD dataset')
    _stimuli_dir = _config['basic']['ANF_path']
    _list_input_files = os.listdir(_stimuli_dir)
    _list_input_files_seed = [x.split('.')[0].split('_')[-1] for x in _list_input_files]
    _dict_df_ITD = {}
    for _seed,_file in zip(_list_input_files_seed,_list_input_files):
        _load_filename =_stimuli_dir+_file
        f = open(_load_filename, 'rb')
        _load_profile_data = pickle.load(f)
        f.close()
        _dict_df_ITD[int(_seed)] = _load_profile_data[-1]
    _df_ITD  = _load_profile_data[-1]
    fs_input_sounds = _load_profile_data[6]
    _sti_duration = np.mean(_df_ITD['End_R'] - _df_ITD['Start_R'] )/fs_input_sounds

    
    # overall interations for ITD curves, peaks, and accuracy 
    print('> start computing ITD tuning curve parameters')
    _df_decoded = []
    for _i_file,_row in list(df_files.iterrows()):
        # load data
        _load_name = import_path+_row['filename']
        print('> processing file:',_load_name)
        with open(_load_name, 'rb') as fp:
            _loaded_list = pickle.load(fp)

        _spike_count_list,acc_set,df_sensitivity = _loaded_list
        _acc,_conf,_df_decoding_error,_MSE,_dict_class_info = acc_set
        _sensitivity_acc = df_sensitivity[df_sensitivity['ITD']==10]['Accuracy'].values[0]
        _df_decoding_error['Matched'] = _df_decoding_error['Target'] == _df_decoding_error['Predicted']
        _acc = _df_decoding_error.Matched.mean()
        _df = _row.copy()
        _df['Accuracy'] = np.round(_acc,5)
        _df['MSE'] = np.round(_MSE,5)
        _df['PairedAcc'] = np.round(_sensitivity_acc,5)

        _df_info = _df.copy()

        ### compute peak information
        _seed = _row['seed']
        _df_spike_count = pd.DataFrame(dict(zip(['SpikeCount','Neuron','Sequence'],_spike_count_list)))
        _df_ITD = _dict_df_ITD[int(_seed)].copy()
        _df_spike_count = _df_spike_count.merge(_df_ITD, on ='Sequence')
        _df_spike_count['ITD']*=1e6
        _df_spike_count['Object'] = _df_spike_count['Neuron'].where(_df_spike_count['Neuron']>=10000,'MSO_L')
        _df_spike_count['Object'] = _df_spike_count['Object'].mask(_df_spike_count['Neuron']>=10000,'MSO_R')

        ## quantile for selecting neurons for curve-plotting
        _quantile_thres = [0.2, 0.8]

        _df_neuron_left = _df_spike_count[_df_spike_count['Object']=='MSO_L'].pivot_table(index = 'Neuron',values='SpikeCount')
        _df_neuron_left = _df_neuron_left[_df_neuron_left['SpikeCount']!=0]
        _lower_quantile = _df_neuron_left.SpikeCount.quantile(_quantile_thres[0])
        _upper_quantile = _df_neuron_left.SpikeCount.quantile(_quantile_thres[1])
        _neuron_list_left = list(_df_neuron_left[(_df_neuron_left.SpikeCount>_lower_quantile)&
                                            (_df_neuron_left.SpikeCount<_upper_quantile)].index)

        _df_neuron_right = _df_spike_count[_df_spike_count['Object']=='MSO_R'].pivot_table(index = 'Neuron',values='SpikeCount')
        _df_neuron_right = _df_neuron_right[_df_neuron_right['SpikeCount']!=0]
        _lower_quantile = _df_neuron_right.SpikeCount.quantile(_quantile_thres[0])
        _upper_quantile = _df_neuron_right.SpikeCount.quantile(_quantile_thres[1])
        _neuron_list_right = list(_df_neuron_right[(_df_neuron_right.SpikeCount>_lower_quantile)&
                                            (_df_neuron_right.SpikeCount<_upper_quantile)].index)

        _neuron_list = _neuron_list_left + _neuron_list_right
        _df_sorted = _df_spike_count[_df_spike_count['Neuron'].isin(_neuron_list)]

        ## get ITD-FR tunning curve 

        _df_ITD_curve = _df_sorted.pivot_table(index = ['Object','ITD'],values='SpikeCount')  
        _df_ITD_curve = pd.DataFrame(_df_ITD_curve.to_records())
        _df_ITD_curve['FiringRate'] = _df_ITD_curve['SpikeCount']/_sti_duration
        for _object in _df_ITD_curve.Object.unique():
            _df_sig = _df_ITD_curve[_df_ITD_curve['Object']==_object]
            _ITD = _df_sig['ITD'].to_numpy()
            _FR = _df_sig['FiringRate'].to_numpy()

            # compute peak parameters
            _peak_ITD, _peak_FR,_half_width = get_peak_parameters_of_ITD_curve(_ITD,_FR)
            _df_peak = _df_info.copy()
            _df_peak['Object'] = _object
            _df_peak['PeakITD'] = _peak_ITD
            _df_peak['PeakHeight'] = _peak_FR
            _df_peak['PeakWidth'] = _half_width
            _df_peak['ITD_v'] =  _ITD
            _df_peak['FR_v'] = _FR
            _df_decoded.append(pd.DataFrame(_df_peak).T)

    _df_decoded = pd.concat(_df_decoded)
    df_curves = _df_decoded.copy()
    
    # export table
    _data_save_name = export_path+'table_ITD_tuning_curves.csv'
    df_curves.to_csv(_data_save_name);
    
    # plot ITD tuning curve
    _cm_1 = cm.PuBu
    _cm_2 = cm.OrRd
    _cm_set = [_cm_1,_cm_2]
    _figsize = [7.5,5]
    _dpi = 150
    _lw = 3
    _fontsize = 18
    plt.rcParams.update({'font.size': _fontsize})
    _df_plot = df_curves.copy()
    _object_set = _df_plot.Object.unique()
    _object_label_mapping = {'MSO_L':'left MSO','MSO_R':'right MSO'}
    _df_plot['ObjectID'] = _df_plot['Object'].map(_object_label_mapping)
    
    
    fig,ax = plt.subplots(figsize=_figsize,dpi = _dpi)
    for _i_object,_object in enumerate(_object_set):
        _df = _df_plot[_df_plot['Object']==_object]
        # averaging smoothed tunning curve 
        _x_stack = []
        _y_stack = []
        for _i_row,_row in _df.iterrows():
            _ITD = _row['ITD_v']
            _FR = _row['FR_v']
            _ITD_interp,_FR_smoothed = interp_and_smooth_ITD_curve(_ITD,_FR,scale = 5)
            _x_stack.append(_ITD_interp)
            _y_stack.append(_FR_smoothed)
        _ITD_mean = np.mean(_x_stack,axis=0)
        _FR_mean = np.mean(_y_stack,axis=0)
        _ITD_mean*=1e-3 # convert from us to ms

        _color = _cm_set[_i_object](0.75)
        _label = _object_label_mapping[_object]
        ax.plot(_ITD_mean,_FR_mean,lw=_lw,c = _color,label = _label)

    ax.set_xlabel('ITD (ms)')
    ax.set_ylabel('Firing rate (sp/s)')
    sns.despine()
    ax.legend(ncol=2,bbox_to_anchor=[0.5,1.25],loc = 10,framealpha=1,edgecolor ='w')

    plt.tight_layout()

    _fig_save_name = export_path+'fig_ITD_tuning_curves.pdf'
    fig.savefig(_fig_save_name,dpi = _dpi);
    
    # plot peak measurements
    _cm_1 = cm.PuBu
    _cm_2 = cm.OrRd
    _cm_set = [_cm_1,_cm_2]
    _figsize = [10,5]
    _dpi = 150
    _lw = 3
    _fontsize = 18
    plt.rcParams.update({'font.size': _fontsize})
    _df_plot = df_curves.copy()
    _object_set = _df_plot.Object.unique()
    _object_label_mapping = {'MSO_L':'left MSO','MSO_R':'right MSO'}
    _df_plot['ObjectID'] = _df_plot['Object'].map(_object_label_mapping)

    fig,axs = plt.subplots(1,3,figsize=_figsize,dpi = _dpi)


    ax = axs[0]
    sns.swarmplot(data = _df_plot,
                  y = 'PeakITD',
                  x = 'ObjectID',
                  hue = 'ObjectID',
                  ax = ax,alpha = 0.7,dodge = True,
                  palette=[x(0.75) for x in _cm_set])
    ax.set_xlabel('Nucleus')
    ax.set_ylabel('Best ITD (us)')
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 60)
    ax.set_xlabel('')
    sns.despine()
    ax.get_legend().remove()

    ax = axs[1]
    sns.swarmplot(data = _df_plot,
                  y = 'PeakHeight',
                  x = 'ObjectID',
                  hue = 'ObjectID',
                  ax = ax,alpha = 0.7,dodge = True,
                  palette=[x(0.75) for x in _cm_set])

    ax.set_ylabel('Peak firing rate (sp/s)')
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 60)
    ax.get_legend().remove()
    ax.set_xlabel('')
    sns.despine()
    ax.set_ylim([0,_df_plot['PeakHeight'].max()*1.2])


    ax = axs[2]
    sns.swarmplot(data = _df_plot,
                  y = 'PeakWidth',
                  x = 'ObjectID',
                  hue = 'ObjectID',
                  ax = ax,alpha = 0.7,dodge = True,
                  palette=[x(0.75) for x in _cm_set])
    ax.set_ylabel('Peak FWHM (us)')
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 60)
    ax.get_legend().remove()
    ax.set_xlabel('')
    sns.despine()
    ax.set_ylim([0,_df_plot['PeakWidth'].max()*1.2])

    plt.tight_layout()
    _fig_save_name = export_path+'fig_ITD_peak_measurements.pdf'
    fig.savefig(_fig_save_name,dpi = _dpi);



