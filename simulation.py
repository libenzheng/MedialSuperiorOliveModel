#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import _pickle as pickle
from datetime import datetime 
from neuron_and_synapse import *
import os
from itertools import product
import brian2 as b
global simulator_name 
simulator_name = 'b'
from datetime import datetime
# for code generation and parallel computing 
# import shutil
# import multiprocessing as mp


def run_all_simulations():
    _config_filename = 'config.pckl'
    with open(_config_filename, 'rb') as _file:
        _config = pickle.load(_file)
    n_seed = _config['stimulation']['n_seed']
    for _seed in range(n_seed):
        print('>> Start simulating seed',_seed+1,' out of ',n_seed)
        build_and_run_single_round_simulation(_seed)
        

def build_and_run_single_round_simulation(_seed = 0):

    function_start_time = datetime.now()

    ## set random seed
    b.seed(_seed)

    ## load config
    _config_filename = 'config.pckl'
    with open(_config_filename, 'rb') as _file:
        _config = pickle.load(_file)
    _dict_config_model = _config['model']
    population_name_set = _dict_config_model['population_name_set'] 
    population_model = _dict_config_model['population_model'] 
    population_neuron_type = _dict_config_model['population_neuron_type'] 
    population_size = _dict_config_model['population_size'] 
    connection_mapping = _dict_config_model['connection_mapping'] 
    import_path = _config['basic']['ANF_path']
    export_path = _config['basic']['data_path']
    os.makedirs(import_path,exist_ok=True)
    os.makedirs(export_path,exist_ok=True)
    switch_multiprocessing = _config['basic']['switch_multiprocessing']
    if switch_multiprocessing:
        import shutil
        import multiprocessing as mp


    ## export_preparation
    _cate_str = 'MSO_SNN'+'_seed_'+str(_seed)
    _codgen_dir = 'cpp_code_'+_cate_str
    _save_name = 'data_'+_cate_str+'.pckl'
    _save_name = export_path+_save_name

    ## input handeling 
    _list_input_files = os.listdir(import_path)
    _list_input_files_seed = [x.split('.')[0].split('_')[-1] for x in _list_input_files]
    _list_input_files_seed = [int(x) for x in _list_input_files_seed]
    _dict_input_files = dict(zip(_list_input_files_seed,_list_input_files))
    _load_filename = import_path+_dict_input_files[_seed]
    print('> loading auditory nerve input:',_load_filename)
    _axis_name = 'ITD(s)'
    with open(_load_filename, 'rb') as _file:
        _load_profile_data = pickle.load(_file)
    ANF_L_input = _load_profile_data[0].spikes
    ANF_R_input = _load_profile_data[1].spikes
    start_position_set_R = _load_profile_data[8]
    ITD_set = _load_profile_data[3]
    population_parameter = [{} for x in population_name_set]
    population_parameter[0] = {'SpikeTrain':ANF_L_input}
    population_parameter[1] = {'SpikeTrain':ANF_R_input}


    # report simulation duration
    fs_input_sounds = _load_profile_data[6]
    df_ITD = _load_profile_data[-1]
    input_duration = int(1000*df_ITD.TrialEnd.max()/fs_input_sounds) # in ms 
    print('> duration to be simulated:',input_duration/1e3,' sec')


    ## setup simulation
    if switch_multiprocessing:
        print('> enable multiprocessing; detected CPU count:',mp.cpu_count())
        #cpp standalone
        b.set_device('cpp_standalone', build_on_run=False)
        #openMP
        n_threads = mp.cpu_count()-1

    #re-arrange information and model
    population_set = {}
    for _i_population,_name_population in enumerate(population_name_set):
        population_set[_name_population] = build_neuron_population(_name_population,
                                                                   _i_population,
                                                                   population_model[_i_population],
                                                                   population_size[_i_population],
                                                                  parameter_assign=population_parameter[_i_population])

    synapse_set = {}
    for _i_synapse,_mapping_synapse in enumerate(connection_mapping):
        synapse_set[_i_synapse] = build_synapse(source_population=population_set[_mapping_synapse[0]],
                                                target_population= population_set[_mapping_synapse[1]],
                                                synapse_model = _mapping_synapse[6],
                                                connect_type = _mapping_synapse[5],
                                                connect_condition = _mapping_synapse[2],
                                                connect_probability = _mapping_synapse[3],
                                                delay = _mapping_synapse[4],
                                               parameter_assign = _mapping_synapse [7])

    ## build network
    # build model with BRAIN2 simulator: Neuron Group
    print('> creating NeuronGroups')
    for _i_population,_population in enumerate(population_set):
        if population_model[_i_population]!='spike_generator':
            _command_set = population_set[_population].generate_variables()
            for _command in _command_set:
                exec(_command)
        else:
            locals().update(population_set[_population].export_variables())
        _command_set = population_set[_population].generate_neuron_population()
        for _command in _command_set:
            exec(_command)
        _command_set = population_set[_population].generate_monitor()
        for _command in _command_set:
            exec(_command)

    # build model with BRAIN2 simulator: Synapses
    print('> creating Synapses')
    for _synapse in synapse_set:
        _command_set = synapse_set[_synapse].generate_variables()
        for _command in _command_set:
            exec(_command)
        _command_set = synapse_set[_synapse].generate_synapse()
        for _command in _command_set:
            exec(_command)
        _command_set = synapse_set[_synapse].generate_connection()
        for _command in _command_set:
            exec(_command)

    # value initialization
    print('> initializing network')
    for _i_population,_population in enumerate(population_set):
        _command = population_set[_population].set_neuron_population_value('v_m','-55.8*b.mV')
        if population_model[_i_population]=='LIF':
            exec(_command[0])



    ## run simulation

    print('> start running simulation')
    simulation_time = input_duration #in msecond
    start_time = datetime.now()
    if switch_multiprocessing:
        print('> generating cpp code')
        b.run(simulation_time*b.ms,report = 'text')
        b.device.build(directory=_codgen_dir, compile=True, run=True, debug=False)
    else:
        b.run(simulation_time*b.ms,report = 'text')
    _elapsed = datetime.now() - start_time
    print('** total time: '+str(_elapsed))

    ## export data
    print('> exporting data')

    _key_set = ['MSO_L','MSO_R']
    dict_spike = {}
    for _key in _key_set:
        _command = 'dict_spike[_key]=SpikeMonitor_'+_key+'.spike_trains()'
        exec(_command)
    # remove units -- default: second
    for _key in dict_spike:
        for _neuron in dict_spike[_key]:
            dict_spike[_key][_neuron]=b.asarray(dict_spike[_key][_neuron])

    _save_obj = [
                    dict_spike,
                    _elapsed,
                    _key_set,
                    start_position_set_R,
                    ITD_set,
                    fs_input_sounds,
                    _seed,
                    df_ITD,
                ]
    with open(_save_name, 'wb') as _file:
        pickle.dump(_save_obj, _file) 
    print('> saved data: '+_save_name)

    # reinit
    if switch_multiprocessing:
        print('> reinitializing simulation core')
        b.device.reinit()
        b.device.activate()
        try:
            shutil.rmtree(_codgen_dir)
        except:
            pass

    print('> complete')

    
