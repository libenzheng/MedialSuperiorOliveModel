#!/usr/bin/env python
# coding: utf-8

import numpy as np
from datetime import datetime
simulator_name = 'b' # for brian2 simulator 

class entity:
    # basic settings and common functions for neurons, synapse and etc.
    def __init__(self,name='unassigned',
             category = 'unassigned',
             seq_index = -1):
        #assigned value
        self.name = name
        self.category = category
        self.seq_index = seq_index #-1 for unassigned
        #auto-generated value
        self.created_time = datetime.now()

    def __call__(self):
        return self.__dict__


    def set_neuron_population_value(self,set_variable,set_value):
        _command_set = []
        _command = 'NeuronGroup_'+self.name+'.'+set_variable+'='+set_value
        _command_set.append(_command)
        return _command_set

        

    def generate_variables(self):
        _command_set = []
        for _key in self.model.parameters.keys():
            if isinstance(self.model.parameters[_key], tuple):            
                _command = '%s=%f'%(_key,self.model.parameters[_key][0])
                if self.model.parameters[_key][1]!='1': #  1 for value without unit
                    _command +='*'+simulator_name+'.'+self.model.parameters[_key][1]
            else:
                _command = ' '
            _command_set.append(_command)
        for _key in self.model.eqs.keys():
            _command = '%s=\'\'\'%s\'\'\''%(_key,self.model.eqs[_key])
            _command_set.append(_command)
        return _command_set

    


    def export_variables(self):
        _export_variable_dict = {}
        for _key in self.model.parameters.keys():
            _export_variable_dict[_key]=self.model.parameters[_key]
        return _export_variable_dict #using  locals().update(mydict) to create variables


    def generate_neuron_population(self):
        _command_set = []
        # neuron population
        if self.model.category[0] == 'N':        
            _key = list(self.model.eqs.keys())
            _command = 'NeuronGroup_'+self.name+'='
            _command += simulator_name+'.NeuronGroup('+str(self.population_size)
            _command += ',model='+_key[0]
            _command += ',threshold='+_key[1]
            _command += ',reset='+_key[2]
            _command += ',refractory='+_key[3]
            _command += ',method ='+_key[4]
            _command += ')'
        
        # spike generator
        if self.model.category[0] == 'G':
            _key = list(self.model.parameters.keys())
            _command = 'NeuronGroup_'+self.name+'='
            _command += simulator_name+'.SpikeGeneratorGroup('+str(self.population_size)
            _command += ',indices='+_key[0]
            _command += ',times='+_key[1]+'*'+simulator_name+'.second'
            _command += ')'
        _command_set.append(_command)
        return _command_set


    def generate_synapse(self):
        _command_set = []
        _key = list(self.model.eqs.keys())
        _source = 'NeuronGroup_'+self.source_population
        _target = 'NeuronGroup_'+self.target_population
        _command = self.name +'='
        _command += simulator_name+'.Synapses('
        _command += _source+','+_target
        _command +=', model ='+_key[0]
        _command +=', on_pre ='+_key[1]+')'
        _command_set.append(_command)
        return _command_set

        

    def generate_connection(self):
        _command_set = []
        if self.connect_condition=='one-to-one':  # updated Apr 30th 2021 for one-to-one mapping of MNTB
            _command = self.name+'.connect(j=\'i\')'
        else:
            _command = self.name+'.connect('
            _command += 'condition=\''+self.connect_condition+'\''
            _command += ',p ='+str(self.connect_probability)+')'
        _command_set.append(_command)

        # w/ SD of 50 us jitter
        _command = self.name+'.delay ='
        _command +='\''
        _command +='(randn()*50 + '  # add SD of delay here 
        _command += str(self.delay[0])
        _command += ')*'+self.delay[1]
        _command +='\''
        _command_set.append(_command)
        return _command_set

    
    def generate_monitor(self,state = 'spike'):
        if state =='spike':
            _command_set = []
            _command = 'SpikeMonitor_'+self.name+'='
            _command += simulator_name+'.SpikeMonitor('
            _command += 'NeuronGroup_'+self.name
            _command += ')'
            _command_set.append(_command)
            return _command_set

        
class build_neuron_population(entity):
    def __init__(self,
                 name,
                 seq_index = -1,
                 neuron_model='unassigned',
                 population_size = -1,
                 parameter_assign=None,
                 eqs_assign = None):

        entity.__init__(self,name,
                        category = 'NeuronPopulation',
                       seq_index = seq_index)

        self.neuron_model = neuron_model
        self.population_size = population_size
        self.parameter_assign = parameter_assign
        self.eqs_assign = eqs_assign


        if self.neuron_model == 'LIF':
            self.model = neuron_model_LIF(self.name,self.parameter_assign,self.eqs_assign)
        elif self.neuron_model == 'spike_generator':
            self.model = neuron_model_spike_generator(self.name,self.parameter_assign)
        else:
            self.model = model()

        
class build_synapse(entity):
    def __init__(self,
                 source_population,
                 target_population,
                 synapse_model,
                parameter_assign =None,
                 eqs_assign = None,
                connect_type = 'exci',
                connect_condition = 'i!=j',
                connect_probability = 0.5,
                delay = (2,'ms')
                ):
        
        self.name = 'Synapse'
        self.name += '_'+source_population.name+'_'+target_population.name
        self.name += '_'+str(source_population.seq_index)+'_'+str(target_population.seq_index)

        self.source_population = source_population.name
        self.target_population = target_population.name
        self.mapping = (source_population.seq_index,target_population.seq_index)
        self.mapping_size = (source_population.population_size,target_population.population_size)

        entity.__init__(self,
                        name=self.name,
                        category = 'Synapse',
                        seq_index = self.mapping)

        self.synapse_model = synapse_model
        self.parameter_assign = parameter_assign
        self.eqs_assign = eqs_assign
        self.connect_type = connect_type
        self.connect_condition = connect_condition
        self.connect_probability = connect_probability
        self.delay = delay

        if self.synapse_model == 'conductance_exp':
            self.model = synapse_model_conductance_exp(self.name,connect_type,self.parameter_assign,self.eqs_assign)
        else:
            self.model = model()


class model:
    def __init__(self,
                 name='unassigned_model',
                 category = 'unassigned_category',
                 parameter_assign = None,
                 eqs_assign = None):
        self.name = name
        self.category = category
        if parameter_assign!=None:
            self.configurate_initial_parameter(parameter_assign)
        if eqs_assign!=None:
            self.configurate_initial_eqs(eqs_assign)
        self.configurate_variable_names()

    def __call__(self):
        return self.__dict__

        
    def configurate_initial_parameter(self,_parameter_updating_dict):
        for _i_key in _parameter_updating_dict.keys():
            if _i_key in self.parameters:
                self.parameters[_i_key]=_parameter_updating_dict[_i_key]

    def configurate_initial_eqs(self,_eqs_updating_dict):
        for _i_key in _eqs_updating_dict.keys():
            if _i_key in self.eqs:
                self.eqs[_i_key]=_eqs_updating_dict[_i_key]

    def configurate_variable_names(self):
        _parameters_new = {}
        for _key in self.parameters.keys():
            _new_key = _key+'_'+ str(self.name)
            _parameters_new[_new_key] = self.parameters[_key]
            for _key_eqs in self.eqs.keys():
                self.eqs[_key_eqs] = self.eqs[_key_eqs].replace(_key,_new_key)

        _eqs_new = {}    
        for _key in self.eqs.keys():
            _new_key = _key+'_'+ str(self.name)
            _eqs_new[_new_key] = self.eqs[_key]
        self.parameters = _parameters_new
        self.eqs = _eqs_new

class neuron_model_LIF(model):
    def __init__(self,name,parameter_assign = None,eqs_assign = None):
        self.name = name
        self.category = 'N_LIF_neuron_model'
        self.parameter_assign = parameter_assign
        self.eqs_assign = eqs_assign
        
        self.eqs = {
            'eqs_model':'''dv_m/dt = (g_l*(E_l-v_m) + g_e*(E_e-v_m) + g_i*(E_i-v_m) + I_ex)/C_m    : volt (unless refractory)
            dg_e/dt = -g_e/tau_e  : siemens   # post-synaptic exc. conductance
            dg_i/dt = -g_i/tau_i  : siemens   # post-synaptic inh. conductance ''',
            'eqs_threshold':'v_m > V_th',
            'eqs_reset':'v_m = V_r',
            'eqs_refractory':'tau_r',
            'eqs_method':'euler'
                    }

        self.parameters = {
                            'C_m':(70,'pF'),
                            'E_l':(-55.8,'mV'),
                            'E_e':(0,'mV'),
                            'E_i':(-70,'mV'),
                            'I_ex':(0,'pA'),
                            #resting setting
                            'V_th':(-50,'mV'),
                            'V_r':(-55.8,'mV'),
                            'tau_r':(2.5,'ms'),
                            'g_l':(13,'nS'),
                            #synapse
                            'tau_e':(0.23,'ms'),
                            'tau_i':(2,'ms')
                        }
        

        model.__init__(self,
                       name=self.name,
                       category = self.category,
                       parameter_assign = self.parameter_assign,
                       eqs_assign = self.eqs_assign)


class neuron_model_spike_generator(model):
    def __init__(self,name,parameter_assign = None):
        self.name = name
        self.category = 'G_spike_generator'
        self.parameter_assign = parameter_assign # spike train data 
        self.parameters = {}
        self.eqs_assign = {}
        self.eqs = {}
        self.parameters['Spike_indices'] = np.array([])
        self.parameters['Spike_times'] = np.array([])
        self.update_input_spike_train(self.parameter_assign)
        model.__init__(self,
                       name=self.name,
                       category = self.category,
                       parameter_assign = self.parameter_assign,
                       eqs_assign = self.eqs_assign)

    def update_input_spike_train(self,parameter_assign):
        if 'SpikeTrain' in parameter_assign:
            processed_spike_train_data = parameter_assign['SpikeTrain']
        else:
            processed_spike_train_data = np.array([])
        _indices_set = np.array([])
        _times_set = np.array([])
        for _i_input_neuron in range(len(processed_spike_train_data)):
            _times = processed_spike_train_data[_i_input_neuron]
            _indices = _i_input_neuron*np.ones(len(_times))
            _indices_set = np.append(_indices_set,_indices)
            _times_set = np.append(_times_set,_times)
        self.parameters['Spike_indices'] = _indices_set
        self.parameters['Spike_times'] = _times_set

            
class synapse_model_conductance_exp(model):
    def __init__(self,name,connect_type = 'exci',parameter_assign = None,eqs_assign = None):
        self.name = name
        self.category = 'S_conductance_exp_synapse_model'
        self.connect_type = connect_type
        self.parameter_assign = parameter_assign
        self.eqs_assign = eqs_assign

        self.parameters = {
                            'w_exci':(15,'nS'),
                            'w_inhi':(75,'nS'),

                            }
        
        eqs_model ={
            'exci': '''
                    #dg_e/dt = -g_e/tau_e  : siemens (clock-driven)  # post-synaptic exc. conductance

                    ''',

            'inhi': '''
                      #dg_i/dt = -g_i/tau_i  : siemens (clock-driven)  # post-synaptic inh. conductance  
                    '''
                     }

        eqs_on_pre = {
            'exci':'g_e += w_exci',
            'inhi':'g_i += w_inhi'
                    }

        self.eqs = {'eqs_model': eqs_model[self.connect_type],
                    'eqs_on_pre':eqs_on_pre[self.connect_type]}

        model.__init__(self,
                       name=self.name,
                       category = self.category,
                       parameter_assign = self.parameter_assign,
                       eqs_assign = self.eqs_assign)





