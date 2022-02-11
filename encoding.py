#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os 
import itertools
import _pickle as pickle
from datetime import datetime 
from scipy.io import wavfile
import scipy.signal as dsp
import cochlea # https://github.com/Jencke/cochlea


def generate_puretone_sound_wave():
    _dict_stimuli = {}
    
    ## load config
    _config_filename = 'config.pckl'
    with open(_config_filename, 'rb') as _file:
        _config = pickle.load(_file)
    _dict_config_puretone = _config['puretone']
    
    fs = _config['stimulation']['fs']
    f_stimuli = _dict_config_puretone['f_stimuli'] 
    phase = _dict_config_puretone['phase'] 
    padding_pre = _dict_config_puretone['padding_pre']
    padding_post = _dict_config_puretone['padding_post']
    ramp = _dict_config_puretone['ramp']
    sound_level_dbspl = _dict_config_puretone['sound_level_dbspl'] 
    wave_duration = _dict_config_puretone['wave_duration']
    #### generate pure tone 
    _t = np.arange(0, wave_duration, 1/fs)
    _signal = np.sin(2 * np.pi * _t * f_stimuli + phase)
    # adjust sound level
    if sound_level_dbspl is not None:
        _signal = get_dbspl(_signal,sound_level_dbspl)
    # add ramping 
    if ramp != 0:
        _ramp_signal = np.linspace(0, 1, int(np.ceil(ramp * fs)))
        _signal[0:len(_ramp_signal)] = _signal[0:len(_ramp_signal)] * _ramp_signal
        _signal[-len(_ramp_signal):] = _signal[-len(_ramp_signal):] * _ramp_signal[::-1]
    # add pading
    _pad_pre_signal = np.zeros(int(padding_pre * fs))
    _pad_post_signal = np.zeros(int(padding_post * fs))
    _signal = np.concatenate((_pad_pre_signal, _signal, _pad_post_signal))
    _t = np.arange(0, len(_signal)/fs, 1/fs)
    _dict_stimuli['PureTone'] = _signal
    
    return _dict_stimuli


def generate_natural_sound_wave():
    
    _dict_stimuli = {}
    
    # load config
    _config_filename = 'config.pckl'
    with open(_config_filename, 'rb') as _file:
        _config = pickle.load(_file)
    _dict_config_wav = _config['wav']
    
    fs = _config['stimulation']['fs']
    padding_pre = _dict_config_wav['padding_pre']
    padding_post = _dict_config_wav['padding_post']
    ramp = _dict_config_wav['ramp']
    sound_level_dbspl = _dict_config_wav['sound_level_dbspl'] 
    wave_duration = _dict_config_wav['wave_duration']
    import_path = _dict_config_wav['import_path']
    os.makedirs(import_path,exist_ok=True)
    
    ## load sound waves 
    _file_list = os.listdir(import_path)
    _wav_list = [x for x in _file_list if x[-4:]=='.wav']
    for _wav_file_name in _wav_list:
        # choose and import data
        _file_name = import_path+_wav_file_name
        _wav_fs, _signal = wavfile.read(_file_name)
        # select first channel 
        _signal = _signal[:,0]
        # chop duration (assuming longer original duration)
        _imported_duration = len(_signal)/_wav_fs
        _signal = _signal[:int(_wav_fs*wave_duration)]
        # resamping the signal
        _signal = resample(_signal, _wav_fs, fs)
        # adjust sound level
        if sound_level_dbspl is not None:
            _signal = get_dbspl(_signal,sound_level_dbspl)
        # add ramping 
        if ramp != 0:
            _ramp_signal = np.linspace(0, 1, int(np.ceil(ramp * fs)))
            _signal[0:len(_ramp_signal)] = _signal[0:len(_ramp_signal)] * _ramp_signal
            _signal[-len(_ramp_signal):] = _signal[-len(_ramp_signal):] * _ramp_signal[::-1]
        # add pading
        _pad_pre_signal = np.zeros(int(padding_pre * fs))
        _pad_post_signal = np.zeros(int(padding_post * fs))
        _signal = np.concatenate((_pad_pre_signal, _signal, _pad_post_signal))
        _t = np.arange(0, len(_signal)/fs, 1/fs)    
        _dict_stimuli[_wav_file_name] = _signal
    
    return _dict_stimuli



def generate_ITD_dataframe(_dict_stimuli,_seed = 0):
    
    print('> generating ITD dataframe')
    
    ## load config
    _config_filename = 'config.pckl'
    with open(_config_filename, 'rb') as _file:
        _config = pickle.load(_file)
    _dict_config_stimulation = _config['stimulation']
    stimuli_type = _config['basic']['stimuli_type']
    
    fs = _dict_config_stimulation['fs']
    _n_replica = _dict_config_stimulation['n_replica']
    _t_inter = _dict_config_stimulation['t_inter']
    _ITD_set = _dict_config_stimulation['ITD_set']

    _ITD_set = np.round(_ITD_set)
    _ITD_set = np.concatenate(([0,],_ITD_set))
    _ITD_set = [x if x < 1000 else 1000 for x in _ITD_set]
    _ITD_set = np.concatenate((-np.flip(_ITD_set)[:-1],_ITD_set))
    _ITD_set = _ITD_set.astype(float)*1e-6
    
    
    ## generate dataset 
    if stimuli_type == 'PureTone':
        _included_stimuli = ['PureTone']
    elif stimuli_type == 'NaturalSound':
        # generate dataset for natural sound stimuli
        _seed_split_dataset = _seed
        _ratio_training = 0.5 
        _df_all_wav_sampels = pd.DataFrame(list(_dict_stimuli.keys()),columns=['Sample'])
        _n_training = np.floor(len(_df_all_wav_sampels)*_ratio_training).astype(int)
        _df_training = _df_all_wav_sampels.sample(_n_training,random_state=_seed_split_dataset)
        _training_stimuli = _df_training['Sample'].to_list()
        _df_testing = _df_all_wav_sampels[~_df_all_wav_sampels['Sample'].isin(_training_stimuli)]
        _df_testing = _df_testing.sample(len(_df_testing),random_state=_seed_split_dataset)
        _testing_stimuli = _df_testing['Sample'].to_list()
        _included_stimuli = _training_stimuli[:_n_replica]
        
    stimuli_dict = {k:_dict_stimuli[k] for k in _dict_stimuli if k in _included_stimuli} 
    stimuli_length_dict = {k:len(_dict_stimuli[k]) for k in stimuli_dict}

    ## build dataframe 
    df_ITD = []
    for _i_replica,_stimulus in itertools.product(range(_n_replica),stimuli_dict):
        _df = pd.DataFrame(_ITD_set,columns=['ITD'])
        _df['Replica'] = _i_replica
        _df['Stimulus'] = _stimulus
        _df['Length'] = _df['Stimulus'].map(stimuli_length_dict)
        _df['TrialLength'] = _df['Length'] + int(_t_inter*fs)
        df_ITD.append(_df)

    df_ITD = pd.concat(df_ITD)
    df_ITD = df_ITD.reset_index(drop=True)

    # shuffling sequence with _seed
    df_ITD['OriginalSequence'] = df_ITD.index
    df_ITD = df_ITD.sample(len(df_ITD),random_state=_seed)
    df_ITD = df_ITD.reset_index(drop=True)
    df_ITD['Sequence'] = df_ITD.index

    # signal start and end timing 
    df_ITD['ITD_shift'] = df_ITD['ITD']*fs

    # fixed right position
    df_ITD['Start_R'] = (df_ITD['TrialLength'] -df_ITD['Length'])/2
    df_ITD['Start_R'] = df_ITD['Start_R'].astype(int)
    df_ITD['End_R'] = df_ITD['Start_R'] + df_ITD['Length']

    # shifted left position
    df_ITD['Start_L'] = df_ITD['Start_R'] + df_ITD['ITD_shift']
    df_ITD['Start_L'] = df_ITD['Start_L'].astype(int)
    df_ITD['End_L'] = df_ITD['End_R'] + df_ITD['ITD_shift']
    df_ITD['End_L'] = df_ITD['End_L'].astype(int)

    # trial end time
    df_ITD['TrialEnd'] = df_ITD['TrialLength'].cumsum()
    df_ITD['TrialStart'] = df_ITD['TrialEnd'] - df_ITD['TrialLength']
    
    # report 
    _report_length = df_ITD.TrialLength.sum()/fs
    print('> Stimulus duration:',_report_length,' sec')
    
    return df_ITD



def generate_spike_train(_dict_stimuli,df_ITD,_seed):
    
    print('> encoding auditory nerve responses')
    
    ## load config
    _config_filename = 'config.pckl'
    with open(_config_filename, 'rb') as _file:
        _config = pickle.load(_file)
    _dict_config_AN = _config['AN']
    
    export_path = _config['basic']['ANF_path']
    f_stimuli = _config['puretone']['f_stimuli']
    fs = _config['stimulation']['fs']
    _ITD_set = _config['stimulation']['ITD_set']

    n_total_ANF = _dict_config_AN['n_population']
    n_group = _dict_config_AN['n_group']
    f_range = _dict_config_AN['f_range']
    fiber_rate_ratio = _dict_config_AN['fiber_rate_ratio']

    f_min = f_stimuli-f_range
    f_max = f_stimuli+f_range
    n_ANF = int(n_total_ANF/n_group)
    n_HSR = int(n_ANF * fiber_rate_ratio[0])
    n_MSR = int(n_ANF * fiber_rate_ratio[1])
    n_LSR = int(n_ANF * fiber_rate_ratio[2])
    
    ## generate sound wave 
    _total_length = df_ITD.TrialLength.sum()
    sti_wave_L = np.zeros(_total_length)
    sti_wave_R = np.zeros(_total_length)
    for _i_row,_row in df_ITD.iterrows():
        _signal = _dict_stimuli[_row['Stimulus']]
        
        _start_R = _row['TrialStart']+_row['Start_R']
        _end_R = _row['TrialStart']+_row['End_R']    
        sti_wave_R[_start_R:_end_R] = _signal

        _start_L = _row['TrialStart']+_row['Start_L']
        _end_L = _row['TrialStart']+_row['End_L']
        sti_wave_L[_start_L:_end_L] = _signal

    ## encoding auditory nerve spike trains
    start_time = datetime.now()
    anf_trains_R = cochlea.run_zilany2014(
                                            sti_wave_R,
                                            fs,
                                            anf_num=(n_HSR,n_MSR,n_LSR),
                                            cf=(f_min,f_max,n_group),
                                            seed=_seed,
                                            # low-freq audible species config for MSO-ITD
                                            species='human' 
                                           )
    
    _elapsed = datetime.now() - start_time
    print('> encoded right-ear responses; time spent: '+str(_elapsed))
    start_time = datetime.now()
    anf_trains_L = cochlea.run_zilany2014(
                                            sti_wave_L,
                                            fs,
                                            anf_num=(n_HSR,n_MSR,n_LSR),
                                            cf=(f_min,f_max,n_group),
                                            seed=_seed,
                                            species='human' 
                                        )

    _elapsed = datetime.now() - start_time
    print('> encoded left-ear responses; time spent: '+str(_elapsed))
    
    ## export model input data
    _save_profile_data = [
                          anf_trains_L,
                          anf_trains_R,
                          f_stimuli,
                          _ITD_set,
                          len(df_ITD),
                          df_ITD.TrialLength[0],
                          fs,
                          (df_ITD['TrialStart']+df_ITD['Start_L']).to_list(),
                          (df_ITD['TrialStart']+df_ITD['Start_R']).to_list(),
                          df_ITD, 
                        ]
    obj = _save_profile_data
    _cate = 'ITD'
    _cate_name = 'ANF_spikes_'+_cate+'_f_'+str(f_stimuli)+'_size_'+str(n_total_ANF)+'_seed_'+str(_seed)
    _save_filename = export_path + _cate_name +'.pckl'
    f = open(_save_filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    print('> saved as',_save_filename)


def generate_auditory_nerve_input():
    # load config
    _config_filename = 'config.pckl'
    with open(_config_filename, 'rb') as _file:
        config = pickle.load(_file)
    import_path = config['basic']['wav_path']
    export_path = config['basic']['ANF_path']
    stimuli_type = config['basic']['stimuli_type']
    os.makedirs(import_path,exist_ok=True)
    os.makedirs(export_path,exist_ok=True)
    n_seed = config['stimulation']['n_seed']
    # generating sound stimuli
    if stimuli_type == 'PureTone':
        print('>> Generating pure tone sound wave')
        _dict_stimuli = generate_puretone_sound_wave()
    elif stimuli_type == 'NaturalSound':
        print('>> Loading natural sound clips')
        _dict_stimuli = generate_natural_sound_wave()
    # generating auditory nerve responses for each random states
    for _seed in range(n_seed):
        print('>> Generating auditory nerve responses; processing seed:',_seed+1,' out of ',n_seed)
        df_ITD = generate_ITD_dataframe(_dict_stimuli,_seed)
        generate_spike_train(_dict_stimuli,df_ITD,_seed)
        

def get_dbspl(signal, dbspl): # from thorns.waves  
    p0 = 20e-6
    rms = np.sqrt(np.sum(signal**2) / signal.size)
    scalled = signal * 10**(dbspl / 20.0) * p0 / rms
    return scalled

def resample(signal, fs, new_fs): # from thorns.waves  
    new_signal = dsp.resample(signal,int(len(signal)*new_fs/fs))
    return new_signal


