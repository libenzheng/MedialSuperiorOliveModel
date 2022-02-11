#!/usr/bin/env python
# coding: utf-8



import _pickle as pickle
import numpy as np
from myelination_and_delay import *

# run this code to update config.pckl file 

######################
#### basic config ####
######################

## basic configuration 
_dict_config = {}
# raw wav files directory
_dict_config['wav_path'] = 'audio/'
# encoded auditory nerve input directory
_dict_config['ANF_path'] = 'input/'
# simulation data export directory
_dict_config['data_path'] = 'data/'
# decoding data export directory
_dict_config['result_path'] = 'result/'
# visualization data export directory
_dict_config['fig_path'] = 'export/'
# stimulation type PureTone or NaturalSound
_dict_config['stimuli_type'] = 'PureTone'
# multiprocessing and cpp standalone code generation
_dict_config['switch_multiprocessing'] = True 
# details in https://brian2.readthedocs.io/en/stable/user/computation.html



#########################
#### encoding config ####
#########################

## puretone encoding 
_dict_config_puretone = {}
# pure-tone frequency (in Hz)
_dict_config_puretone['f_stimuli'] = 300
# phase (in radians)
_dict_config_puretone['phase'] = 0
# padding silence at begaining and ending (in secound )
_dict_config_puretone['padding_pre'] = 0.05
_dict_config_puretone['padding_post'] = 0.05
# ramping duration
_dict_config_puretone['ramp'] = 0.02
# sound level in dbspl
_dict_config_puretone['sound_level_dbspl'] = 50
# duration
_dict_config_puretone['wave_duration'] = 0.1





## natural sound encoding
_dict_config_wav = {}
# directory wav files
_dict_config_wav['import_path'] = 'audio/'
# padding silence at begaining and ending (in secound )
_dict_config_wav['padding_pre'] = 0.05
_dict_config_wav['padding_post'] = 0.05
# ramping duration
_dict_config_wav['ramp'] = 0.02
# sound level in dbspl
_dict_config_wav['sound_level_dbspl'] = 50
# duration
_dict_config_wav['wave_duration'] = 1





## stimulation config
_dict_config_stimulation = {}
# number of random seeds (permutations)
_dict_config_stimulation['n_seed'] = 2
# number of replication per stimuli
_dict_config_stimulation['n_replica'] = 10 
# inter-stimulus interval 
_dict_config_stimulation['t_inter'] = 0.1
# set ITD set in us (include 0 ITD by default); positive ITD refers to leading sound at right ear
_dict_config_stimulation['ITD_set'] = [10,20,40,80,100,150,200,250,300,350,400,450,500,600,800,1000]
# sound wave sampling frequency (in Hz)
_dict_config_stimulation['fs'] = 100e3





## auditory nerve config
_dict_config_AN = {}
# number of nerves 
_dict_config_AN['n_population'] = 1000
# number of characteristic frequencies 
_dict_config_AN['n_group'] = 1
# range of characteristic frequencies 
_dict_config_AN['f_range'] = 0
# ratio of spontaneous rate fibers [HSR,MSR,LSR]
_dict_config_AN['fiber_rate_ratio'] = [0.6,0.2,0.2]



##########################
#### MSO model config ####
##########################

### MSO model config 
_dict_config_model = {}

## neural population configuration

# population name
population_name_set = [
                       'ANF_L','ANF_R',                          # Auditory neural fiber (ANF)
                       'SBC_L','SBC_R','GBC_L','GBC_R',          # Anteroventral cochlear nucleus (AVCN)
                       'LNTB_L','LNTB_R','MNTB_L','MNTB_R',      # Trapezoid body (TB)
                       'MSO_L','MSO_R'                           # Medial superior olive (SO)
                      ]
# population model 
population_model = [
                    'spike_generator','spike_generator',
                    'LIF','LIF','LIF','LIF',
                    'LIF','LIF','LIF','LIF',
                    'LIF','LIF',
                   ]
# population neuron type
population_neuron_type = [ 
                            'exci','exci',
                            'exci','exci','exci','exci',
                            'inhi','inhi','inhi','inhi',
                            'exci','exci'
                         ]

# population size
n_population = 1000
population_size = n_population* np.ones(len(population_name_set))


## synapse configuration

# connectivity 
p_ANF2AVCN = 40/n_population
p_SBC2MSO = 6/n_population
p_GBC2MNTB = 1/n_population
p_MNTB2MSO = 3/n_population
p_GBC2LNTB = 3/n_population
p_LNTB2MSO = 3/n_population

# connection config
condition = 'i!=j'
synapse_model = 'conductance_exp'
parameter_default =None
delay = (600,'us') 

# specialized config 
condition_GBC2MNTB = 'one-to-one' # calyx of held
parameter_GBC2MNTB = {'w_exci':(250,'nS')} 
delay_MNTB_con = (400,'us')
delay_SBC_ipsi = (1200,'us')
delay_LNTB_ipsi = (400,'us')

## myelination config 

# myelin thickness
SBC_myelin_thickness = 0.25 # SBC to contralateral MSO in um
GBC_myelin_thickness = 0.40 # GBC to contralateral MNTB in um
# geometric parameters in um
SBC_fiber_length = 4500
GBC_fiber_length = 4500 
SBC_heminode_length = 500
GBC_heminode_length = 700
# synaptic delays in us
SBC_synaptic_delay = 600
GBC_synaptic_delay = 200 # calyx of held
# conduction velocity in m/s
SBC_cv = get_conduction_velocity(SBC_myelin_thickness)
GBC_cv = get_conduction_velocity(GBC_myelin_thickness)
# overal delay
delay_SBC = (get_delay(SBC_cv,SBC_fiber_length,SBC_heminode_length,SBC_synaptic_delay),'us') 
delay_GBC = (get_delay(GBC_cv,GBC_fiber_length,GBC_heminode_length,GBC_synaptic_delay),'us') 


## circuitry config

connection_mapping = [
                        # ANF
                        ('ANF_L','SBC_L',condition,p_ANF2AVCN,delay,'exci',synapse_model,parameter_default),
                        ('ANF_L','GBC_L',condition,p_ANF2AVCN,delay,'exci',synapse_model,parameter_default),
                        ('ANF_R','SBC_R',condition,p_ANF2AVCN,delay,'exci',synapse_model,parameter_default),
                        ('ANF_R','GBC_R',condition,p_ANF2AVCN,delay,'exci',synapse_model,parameter_default),
                        # SBC
                        ('SBC_L','MSO_L',condition,p_SBC2MSO,delay_SBC_ipsi,'exci',synapse_model,parameter_default),
                        ('SBC_L','MSO_R',condition,p_SBC2MSO,delay_SBC,'exci',synapse_model,parameter_default),
                        ('SBC_R','MSO_R',condition,p_SBC2MSO,delay_SBC_ipsi,'exci',synapse_model,parameter_default),
                        ('SBC_R','MSO_L',condition,p_SBC2MSO,delay_SBC,'exci',synapse_model,parameter_default),
                        # GBC
                        ('GBC_L','LNTB_L',condition,p_GBC2LNTB,delay,'exci',synapse_model,parameter_default),
                        ('GBC_L','MNTB_R',condition_GBC2MNTB,p_GBC2MNTB,delay_GBC,'exci',synapse_model,parameter_GBC2MNTB),
                        ('GBC_R','LNTB_R',condition,p_GBC2LNTB,delay,'exci',synapse_model,parameter_default),
                        ('GBC_R','MNTB_L',condition_GBC2MNTB,p_GBC2MNTB,delay_GBC,'exci',synapse_model,parameter_GBC2MNTB),
                        # LNTB
                        ('LNTB_L','MSO_L',condition,p_LNTB2MSO,delay_LNTB_ipsi,'inhi',synapse_model,parameter_default),
                        ('LNTB_R','MSO_R',condition,p_LNTB2MSO,delay_LNTB_ipsi,'inhi',synapse_model,parameter_default),
                        # MNTB
                        ('MNTB_L','MSO_L',condition,p_MNTB2MSO,delay_MNTB_con,'inhi',synapse_model,parameter_default),
                        ('MNTB_R','MSO_R',condition,p_MNTB2MSO,delay_MNTB_con,'inhi',synapse_model,parameter_default),
                    ]


## save
_dict_config_model['population_name_set'] = population_name_set
_dict_config_model['population_model'] = population_model
_dict_config_model['population_neuron_type'] = population_neuron_type
_dict_config_model['population_size'] = population_size
_dict_config_model['connection_mapping'] = connection_mapping




#####################
#### save config ####
#####################
dict_save_config = {
                    'basic':_dict_config,
                    'puretone':_dict_config_puretone,
                    'wav':_dict_config_wav,
                    'stimulation':_dict_config_stimulation,
                    'AN':_dict_config_AN,
                    'model':_dict_config_model,
                    }

_save_filename = 'config.pckl'
with open(_save_filename, 'wb') as _file:
    pickle.dump(dict_save_config, _file)

    




